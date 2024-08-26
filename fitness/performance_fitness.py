import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def calculate_metrics(assay_scores: np.ndarray, model_scores: np.ndarray) -> Dict[str, float]:
    """Calculate Spearman correlation, AUC, and MCC."""
    spearman_corr = stats.spearmanr(assay_scores, model_scores).correlation
    auc = roc_auc_score(y_true=(assay_scores > np.median(assay_scores)).astype(int), y_score=model_scores)
    mcc = matthews_corrcoef(y_true=(assay_scores > np.median(assay_scores)).astype(int), 
                            y_pred=(model_scores > np.median(model_scores)).astype(int))
    return {
        "Spearman": abs(spearman_corr),
        "AUC": max(auc,1-auc),
        "MCC": abs(mcc)
    }

def get_performance_dataset(df: pd.DataFrame, assay_col: str, score_columns: List[str]) -> Dict[str, Dict[str, float]]:
    """Process a single dataset and return metrics for all models."""
    results = {}
    df = df.dropna(subset=['mutant'])  # Drop WT sequences if present
    
    for model in score_columns:
        subset = df.dropna(subset=[assay_col, model])
        
        if subset.empty:
            results[model] = {metric: np.nan for metric in ['Spearman', 'AUC', 'MCC']}
        else:
            try:
                results[model] = calculate_metrics(np.array(subset[assay_col]), np.array(subset[model]))
            except Exception as e:
                print(f"Error calculating metrics for {model}: {str(e)}")
                print(f"Sample data - {assay_col}: {subset[assay_col].head()}, {model}: {subset[model].head()}")
                results[model] = {metric: np.nan for metric in ['Spearman', 'AUC', 'MCC']}
    
    return results

def bootstrap_se(data: pd.DataFrame, group_col: str, metric_col: str, types: List[str], score_columns: List[str], number_assay_reshuffle: int = 10000) -> Dict[str, float]:
    # Identify the overall best model for each type and for 'All'
    best_models = data.groupby(group_col)[score_columns].mean().idxmax(axis=1)
    best_model_all = data[score_columns].mean().idxmax()
    
    bootstrap_means = {type_: [] for type_ in types + ['All']}
    
    for _ in range(number_assay_reshuffle):
        resampled = data.sample(frac=1.0, replace=True)
        resampled_groups = resampled.groupby(group_col)
        resampled_averages = {}
        
        for type_ in types:
            if type_ in resampled_groups.groups:
                group_data = resampled_groups.get_group(type_)
                if not group_data.empty:
                    best_model_for_type = best_models[type_]
                    # Calculate differences from the best model for this type
                    diffs = group_data[metric_col] - group_data[best_model_for_type]
                    resampled_averages[type_] = diffs.mean()
                    bootstrap_means[type_].append(resampled_averages[type_])
                else:
                    resampled_averages[type_] = np.nan
            else:
                resampled_averages[type_] = np.nan
        
        # Handle 'All' category
        all_diffs = resampled[metric_col] - resampled[best_model_all]
        bootstrap_means['All'].append(all_diffs.mean())

    se = {}
    for type_ in types + ['All']:
        values = [v for v in bootstrap_means[type_] if not np.isnan(v)]
        if len(values) > 1:
            se[type_] = np.std(values, ddof=1)
        else:
            se[type_] = np.nan

    return se

def calculate_RNA_types_averages_with_se(wt_seqs: pd.DataFrame, types: List[str], score_columns: List[str], calculate_se: bool, number_assay_reshuffle: int = 100) -> pd.DataFrame:
    """Calculate average metrics and optionally bootstrap standard errors for each type and model."""
    metrics = ['Spearman', 'AUC', 'MCC']
    result_data = []

    for model in score_columns:
        model_data = {'Model': model}
        for metric in metrics:
            metric_col = f'{metric}_{model}'
            for rna_type in types + ['All']:
                if rna_type == 'All':
                    mean_value = wt_seqs[metric_col].mean()
                else:
                    mean_value = wt_seqs[wt_seqs['RNA_TYPE'] == rna_type][metric_col].mean()
                
                model_data[f'{metric}_{rna_type}_Mean'] = mean_value
                
            if calculate_se:
                metric_columns = [f'{metric}_{col}' for col in score_columns]
                se = bootstrap_se(wt_seqs, 'RNA_TYPE', metric_col, types, metric_columns, number_assay_reshuffle)
                for rna_type in types + ['All']:
                    model_data[f'{metric}_{rna_type}_SE'] = se[rna_type]
        
        result_data.append(model_data)

    result_df = pd.DataFrame(result_data)
    
    # Reorder columns
    column_order = ['Model']
    for metric in metrics:
        for rna_type in types + ['All']:
            column_order.append(f'{metric}_{rna_type}_Mean')
            if calculate_se:
                column_order.append(f'{metric}_{rna_type}_SE')
    
    result_df = result_df[column_order]
    
    return result_df

def calculate_mutation_depth_averages_with_se(wt_seqs: pd.DataFrame, score_columns: List[str], combined_dir: str, calculate_se: bool, number_assay_reshuffle: int = 10000) -> pd.DataFrame:
    """Calculate average metrics and optionally bootstrap standard errors for single and multiple mutations."""
    for _, row in wt_seqs.iterrows():
        dataset = row['DMS_ID']
        df_path = f'{combined_dir}/{dataset}.csv'
        
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            if 'Depth' not in df:
                assay_col = 'DMS_score'
                mutation_column = 'mutant'
                df[mutation_column] = df[mutation_column].astype(str)
                df['mutation_count'] = df[mutation_column].apply(lambda x: len(x.split(',')))
                df['Depth'] = np.where(df['mutation_count'] == 1, 'single', 'multiple')
                df.to_csv(df_path, index=False)
            
            for depth in ['single', 'multiple']:
                df_subset_depth = df[df['Depth'] == depth]
                dataset_results = get_performance_dataset(df_subset_depth, 'DMS_score', score_columns)
                for model in score_columns:
                    for metric in ['Spearman', 'AUC', 'MCC']:
                        wt_seqs.loc[wt_seqs['DMS_ID'] == dataset, f'{metric}_{model}_{depth}'] = dataset_results[model][metric]

    # Calculate averages and bootstrap SE
    average_metrics = {model: {} for model in score_columns}
    se_metrics = {model: {} for model in score_columns} if calculate_se else None
    
    for model in score_columns:
        for metric in ['Spearman', 'AUC', 'MCC']:
            average_metrics[model][metric] = {}
            if calculate_se:
                se_metrics[model][metric] = {}
            for depth in ['single', 'multiple']:
                values = wt_seqs[f'{metric}_{model}_{depth}']
                average_metrics[model][metric][depth] = np.nanmean(values)
            
            # Calculate 'All' category
            all_values = pd.concat([wt_seqs[f'{metric}_{model}_single'], wt_seqs[f'{metric}_{model}_multiple']])
            average_metrics[model][metric]['All'] = np.nanmean([average_metrics[model][metric]['single'], average_metrics[model][metric]['multiple']])
            
            if calculate_se:
                for depth in ['single', 'multiple']:
                    metric_columns = [f'{metric}_{col}_{depth}' for col in score_columns]
                    se = bootstrap_se(wt_seqs, 'DMS_ID', f'{metric}_{model}_{depth}', [depth], metric_columns, number_assay_reshuffle)
                    se_metrics[model][metric][depth] = se[depth]
                # For 'All', we need to combine the SE of 'single' and 'multiple'
                se_single = se_metrics[model][metric]['single']
                se_multiple = se_metrics[model][metric]['multiple']
                se_metrics[model][metric]['All'] = np.sqrt((se_single**2 + se_multiple**2) / 2)  # Average of variances

    # Prepare the result DataFrame
    data = []
    for depth in ['single', 'multiple', 'All']:
        for model in score_columns:
            row = {'Depth': depth, 'Model': model}
            for metric in ['Spearman', 'AUC', 'MCC']:
                row[f'{metric}_Mean'] = average_metrics[model][metric][depth]
                if calculate_se:
                    row[f'{metric}_SE'] = se_metrics[model][metric][depth]
            data.append(row)

    result_df = pd.DataFrame(data)
    result_df = result_df.sort_values('Depth')
    
    return result_df

def analyze_datasets(wt_seqs: pd.DataFrame, combined_dir: str, score_columns: List[str]) -> pd.DataFrame:
    """Analyze all datasets and return results for all models."""
    for _, row in wt_seqs.iterrows():
        dataset = row['DMS_ID']
        df_path = f'{combined_dir}/{dataset}.csv'
        
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            assay_col = 'DMS_score'
            dataset_results = get_performance_dataset(df, assay_col, score_columns)
            
            for model in score_columns:
                for i, metric in enumerate(['Spearman', 'AUC', 'MCC']):
                    wt_seqs.loc[wt_seqs['DMS_ID'] == dataset, f'{metric}_{model}'] = dataset_results[model][metric]
    
    return wt_seqs

def save_assay_level_results(wt_seqs: pd.DataFrame, score_columns: List[str], output_file: str):
    """Save assay-level results with one row per assay, per model."""
    data = []
    for _, row in wt_seqs.iterrows():
        for model in score_columns:
            data.append({
                'DMS_ID': row['DMS_ID'],
                'RNA_TYPE': row['RNA_TYPE'],
                'Model': model,
                'Spearman': row[f'Spearman_{model}'],
                'AUC': row[f'AUC_{model}'],
                'MCC': row[f'MCC_{model}']
            })
    
    assay_df = pd.DataFrame(data)
    assay_df.to_csv(output_file, index=False)

def main(args):
    wt_seqs = pd.read_csv(args.reference_file)
    score_columns = ['Evo_score','GenSLM_score','Nucleotide_Transformer_score','RNA_FM_score','RiNALMo_score']
    wt_seqs = analyze_datasets(wt_seqs, args.combined_dir, score_columns)
    types = ['mRNA', 'tRNA', 'Aptamer', 'Ribozyme']
    
    # Calculate averages and standard errors per RNA type
    result_df_type = calculate_RNA_types_averages_with_se(wt_seqs, types, score_columns, args.calculate_se)
    
    # Calculate averages and standard errors per mutation depth
    result_df_mutation_depth = calculate_mutation_depth_averages_with_se(wt_seqs, score_columns, args.combined_dir, False)

    # Ensure the performance directory exists
    os.makedirs('./performance', exist_ok=True)

    # Save results to CSV
    result_df_type.to_csv('./performance/results_by_rna_type.csv', index=False)
    result_df_mutation_depth.to_csv('./performance/results_by_mutation_depth.csv', index=False)
    
    # Save assay-level results
    save_assay_level_results(wt_seqs, score_columns, './performance/assay_level_results.csv')

    # Print results
    print("\nMetrics by RNA Type:")
    print(result_df_type)
    print("\nMetrics by Mutation Depth:")
    print(result_df_mutation_depth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze RNA datasets and calculate metrics for multiple models.")
    parser.add_argument('--reference_file', type=str, default='reference_sheet_cleaned_CAS.csv',
                        help='Path to the reference CSV file')
    parser.add_argument('--combined_dir', type=str, default='combined_results',
                        help='Base directory for combined result files')
    parser.add_argument('--calculate_se', action='store_true',
                        help='Calculate standard errors (computationally intensive)')
    args = parser.parse_args()
    main(args)