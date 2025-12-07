#!/usr/bin/env python3

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


def calculate_metrics(
    assay_scores: np.ndarray, model_scores: np.ndarray
) -> Dict[str, float]:
    """Calculate Spearman correlation, AUC, and MCC."""
    spearman_corr = stats.spearmanr(assay_scores, model_scores).correlation
    auc = roc_auc_score(
        y_true=(assay_scores > np.median(assay_scores)).astype(int),
        y_score=model_scores,
    )
    mcc = matthews_corrcoef(
        y_true=(assay_scores > np.median(assay_scores)).astype(int),
        y_pred=(model_scores > np.median(model_scores)).astype(int),
    )
    return {"Spearman": abs(spearman_corr), "AUC": max(auc, 1 - auc), "MCC": abs(mcc)}


def get_performance_dataset(
    df: pd.DataFrame, assay_col: str, score_columns: List[str]
) -> Dict[str, Dict[str, float]]:
    """Process a single dataset and return metrics for all models."""
    results = {}
    df = df.dropna(subset=["mutant"])  # Drop WT sequences if present

    for model in score_columns:
        try:
            subset = df.dropna(subset=[assay_col, model])
        except:
            subset = None

        if subset is None or subset.empty:
            results[model] = {metric: np.nan for metric in ["Spearman", "AUC", "MCC"]}
        else:
            try:
                results[model] = calculate_metrics(
                    np.array(subset[assay_col]), np.array(subset[model])
                )
            except Exception as e:
                print(f"Error calculating metrics for {model}: {str(e)}")
                print(
                    f"Sample data - {assay_col}: {subset[assay_col].head()}, {model}: {subset[model].head()}"
                )
                results[model] = {
                    metric: np.nan for metric in ["Spearman", "AUC", "MCC"]
                }

    return results


def bootstrap_se(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    types: List[str],
    score_columns: List[str],
    number_assay_reshuffle: int = 10000,
) -> Dict[str, float]:
    # Identify the overall best model for each type and for 'All'
    best_models = data.groupby(group_col)[score_columns].mean().idxmax(axis=1)
    best_model_all = data[score_columns].mean().idxmax()

    bootstrap_means = {type_: [] for type_ in types + ["All"]}

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
        bootstrap_means["All"].append(all_diffs.mean())

    se = {}
    for type_ in types + ["All"]:
        values = [v for v in bootstrap_means[type_] if not np.isnan(v)]
        if len(values) > 1:
            se[type_] = np.std(values, ddof=1)
        else:
            se[type_] = np.nan

    return se


def bootstrap_se_of_means(
    df: pd.DataFrame,
    type_col: str,
    metric_col: str,
    types: List[str],
    metric_columns: List[str],
    n_iterations: int,
) -> float:
    """
    Calculate the standard error of the mean of type means using bootstrapping.
    Takes into account assay reshuffling and handles maximum scores correctly.

    Args:
        df: DataFrame containing the data
        type_col: Column name containing RNA types
        metric_col: Column name of the metric to calculate SE for
        types: List of RNA types to include
        metric_columns: List of metric columns (used for assay reshuffling)
        n_iterations: Number of bootstrap iterations

    Returns:
        float: Standard error of the mean of type means
    """
    bootstrap_means = []

    # Get the true values for comparison
    true_type_means = {
        rna_type: df[df[type_col] == rna_type][metric_col].mean() for rna_type in types
    }
    true_mean = sum(true_type_means.values()) / len(true_type_means)

    # If this is the maximum scoring model for this metric,
    # return 0 as the standard error
    other_scores = df[metric_columns].mean()
    if all(other_scores[metric_col] >= other_scores):
        return 0.0

    for _ in range(n_iterations):
        # For each iteration, first reshuffle the assay scores
        reshuffled_df = df.copy()
        metric_data = reshuffled_df[metric_columns].values
        np.random.shuffle(metric_data)
        reshuffled_df[metric_columns] = metric_data

        # Then calculate means for each type
        type_means = []
        for rna_type in types:
            type_data = reshuffled_df[reshuffled_df[type_col] == rna_type][metric_col]
            if len(type_data) > 0:
                # For each type, bootstrap sample with replacement
                bootstrap_sample = type_data.sample(n=len(type_data), replace=True)
                type_means.append(bootstrap_sample.mean())

        # Calculate mean of type means for this iteration
        if type_means:
            bootstrap_means.append(sum(type_means) / len(type_means))

    # Calculate standard error from bootstrap distribution
    return np.std(bootstrap_means)


def calculate_RNA_types_averages_with_se(
    wt_seqs: pd.DataFrame,
    types: List[str],
    score_columns: List[str],
    calculate_se: bool,
    number_assay_reshuffle: int = 1000,
) -> pd.DataFrame:
    """Calculate average metrics and optionally bootstrap standard errors for each type and model."""
    metrics = ["Spearman", "AUC", "MCC"]
    result_data = []

    for model in score_columns:
        model_data = {"Model": model}
        for metric in metrics:
            metric_col = f"{metric}_{model}"
            type_means = {}
            for rna_type in types:
                mean_value = wt_seqs[wt_seqs["RNA_TYPE"] == rna_type][metric_col].mean()
                type_means[rna_type] = mean_value
                model_data[f"{metric}_{rna_type}_Mean"] = mean_value

            # Calculate 'All' as average of type means
            all_mean = sum(type_means.values()) / len(type_means)
            model_data[f"{metric}_All_Mean"] = all_mean

            if calculate_se:
                metric_columns = [f"{metric}_{col}" for col in score_columns]
                type_se = bootstrap_se(
                    wt_seqs,
                    "RNA_TYPE",
                    metric_col,
                    types,
                    metric_columns,
                    number_assay_reshuffle,
                )
                for rna_type in types + ["All"]:
                    model_data[f"{metric}_{rna_type}_SE"] = type_se[rna_type]

        result_data.append(model_data)

    result_df = pd.DataFrame(result_data)

    # Reorder columns
    column_order = ["Model"]
    for metric in metrics:
        for rna_type in types + ["All"]:
            column_order.append(f"{metric}_{rna_type}_Mean")
            if calculate_se:
                column_order.append(f"{metric}_{rna_type}_SE")

    result_df = result_df[column_order]

    return result_df


def calculate_mutation_depth_averages_with_se(
    wt_seqs: pd.DataFrame,
    score_columns: List[str],
    combined_dir: str,
    calculate_se: bool,
    number_assay_reshuffle: int = 10000,
) -> pd.DataFrame:
    """Calculate average metrics and optionally bootstrap standard errors for single and multiple mutations."""
    for _, row in wt_seqs.iterrows():
        dataset = row["DMS_ID"]
        df_path = f"{combined_dir}/{dataset}.csv"

        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            if "Depth" not in df:
                assay_col = "DMS_score"
                mutation_column = "mutant"
                df[mutation_column] = df[mutation_column].astype(str)
                df["mutation_count"] = df[mutation_column].apply(
                    lambda x: len(x.split(","))
                )
                df["Depth"] = np.where(df["mutation_count"] == 1, "single", "multiple")
                df.to_csv(df_path, index=False)

            for depth in ["single", "multiple"]:
                df_subset_depth = df[df["Depth"] == depth]
                dataset_results = get_performance_dataset(
                    df_subset_depth, "DMS_score", score_columns
                )
                for model in score_columns:
                    for metric in ["Spearman", "AUC", "MCC"]:
                        wt_seqs.loc[
                            wt_seqs["DMS_ID"] == dataset, f"{metric}_{model}_{depth}"
                        ] = dataset_results[model][metric]

    # Calculate averages and bootstrap SE
    average_metrics = {model: {} for model in score_columns}
    se_metrics = {model: {} for model in score_columns} if calculate_se else None

    for model in score_columns:
        for metric in ["Spearman", "AUC", "MCC"]:
            average_metrics[model][metric] = {}
            if calculate_se:
                se_metrics[model][metric] = {}
            for depth in ["single", "multiple"]:
                values = wt_seqs[f"{metric}_{model}_{depth}"]
                average_metrics[model][metric][depth] = np.nanmean(values)

            # Calculate 'All' category
            all_values = pd.concat(
                [
                    wt_seqs[f"{metric}_{model}_single"],
                    wt_seqs[f"{metric}_{model}_multiple"],
                ]
            )
            average_metrics[model][metric]["All"] = np.nanmean(
                [
                    average_metrics[model][metric]["single"],
                    average_metrics[model][metric]["multiple"],
                ]
            )

            if calculate_se:
                for depth in ["single", "multiple"]:
                    metric_columns = [
                        f"{metric}_{col}_{depth}" for col in score_columns
                    ]
                    se = bootstrap_se(
                        wt_seqs,
                        "DMS_ID",
                        f"{metric}_{model}_{depth}",
                        [depth],
                        metric_columns,
                        number_assay_reshuffle,
                    )
                    se_metrics[model][metric][depth] = se[depth]
                # For 'All', we need to combine the SE of 'single' and 'multiple'
                se_single = se_metrics[model][metric]["single"]
                se_multiple = se_metrics[model][metric]["multiple"]
                se_metrics[model][metric]["All"] = np.sqrt(
                    (se_single**2 + se_multiple**2) / 2
                )  # Average of variances

    # Prepare the result DataFrame
    data = []
    for depth in ["single", "multiple", "All"]:
        for model in score_columns:
            row = {"Depth": depth, "Model": model}
            for metric in ["Spearman", "AUC", "MCC"]:
                row[f"{metric}_Mean"] = average_metrics[model][metric][depth]
                if calculate_se:
                    row[f"{metric}_SE"] = se_metrics[model][metric][depth]
            data.append(row)

    result_df = pd.DataFrame(data)
    result_df = result_df.sort_values("Depth")

    return result_df


def analyze_datasets(
    wt_seqs: pd.DataFrame, combined_dir: str, score_columns: List[str]
) -> pd.DataFrame:
    """Analyze all datasets and return results for all models."""
    for _, row in wt_seqs.iterrows():
        dataset = row["DMS_ID"]
        df_path = f"{combined_dir}/{dataset}.csv"

        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            assay_col = "DMS_score"
            dataset_results = get_performance_dataset(df, assay_col, score_columns)

            for model in score_columns:
                for i, metric in enumerate(["Spearman", "AUC", "MCC"]):
                    wt_seqs.loc[wt_seqs["DMS_ID"] == dataset, f"{metric}_{model}"] = (
                        dataset_results[model][metric]
                    )

    return wt_seqs


def save_assay_level_results(
    wt_seqs: pd.DataFrame, score_columns: List[str], output_file: str
):
    """Save assay-level results with one row per assay, per model."""
    data = []
    for _, row in wt_seqs.iterrows():
        for model in score_columns:
            data.append(
                {
                    "DMS_ID": row["DMS_ID"],
                    "RNA_TYPE": row["RNA_TYPE"],
                    "Model": model,
                    "Spearman": row[f"Spearman_{model}"],
                    "AUC": row[f"AUC_{model}"],
                    "MCC": row[f"MCC_{model}"],
                }
            )

    assay_df = pd.DataFrame(data)
    assay_df.to_csv(output_file, index=False)


def calculate_combined_averages_with_se(
    wt_seqs: pd.DataFrame,
    types: List[str],
    score_columns: List[str],
    combined_dir: str,
    calculate_se: bool,
    number_assay_reshuffle: int = 10000,
) -> pd.DataFrame:
    """Calculate average metrics for combinations of RNA type and mutation depth."""
    # First, ensure mutation depth columns are added to the dataframe
    for _, row in wt_seqs.iterrows():
        dataset = row["DMS_ID"]
        df_path = f"{combined_dir}/{dataset}.csv"

        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            if "Depth" not in df:
                mutation_column = "mutant"
                df[mutation_column] = df[mutation_column].astype(str)
                df["mutation_count"] = df[mutation_column].apply(
                    lambda x: len(x.split(","))
                )
                df["Depth"] = np.where(df["mutation_count"] == 1, "single", "multiple")
                df.to_csv(df_path, index=False)

            # Calculate metrics for each depth and RNA type combination
            for depth in ["single", "multiple"]:
                df_subset_depth = df[df["Depth"] == depth]
                if not df_subset_depth.empty:
                    dataset_results = get_performance_dataset(
                        df_subset_depth, "DMS_score", score_columns
                    )
                    for model in score_columns:
                        for metric in ["Spearman", "AUC", "MCC"]:
                            wt_seqs.loc[
                                wt_seqs["DMS_ID"] == dataset,
                                f"{metric}_{model}_{depth}",
                            ] = dataset_results[model][metric]

            # Also calculate metrics for all mutations (without filtering by depth)
            dataset_results_all = get_performance_dataset(
                df, "DMS_score", score_columns
            )
            for model in score_columns:
                for metric in ["Spearman", "AUC", "MCC"]:
                    wt_seqs.loc[
                        wt_seqs["DMS_ID"] == dataset, f"{metric}_{model}_overall"
                    ] = dataset_results_all[model][metric]

    # Calculate averages for each combination
    results = []
    for model in score_columns:
        for depth in ["single", "multiple", "overall"]:
            # First calculate per RNA type
            for rna_type in types:
                row_data = {"Model": model, "Depth": depth, "RNA_TYPE": rna_type}

                type_datasets = wt_seqs[wt_seqs["RNA_TYPE"] == rna_type][
                    "DMS_ID"
                ].tolist()
                type_subset = wt_seqs[wt_seqs["DMS_ID"].isin(type_datasets)]

                for metric in ["Spearman", "AUC", "MCC"]:
                    col_name = f"{metric}_{model}_{depth}"
                    values = type_subset[col_name].dropna()

                    if len(values) > 0:
                        row_data[f"{metric}_Mean"] = np.mean(values)

                        if calculate_se:
                            # Calculate SE using bootstrap
                            metric_columns = [
                                f"{metric}_{col}_{depth}" for col in score_columns
                            ]
                            se = bootstrap_se(
                                type_subset,
                                "DMS_ID",
                                col_name,
                                [rna_type],
                                metric_columns,
                                number_assay_reshuffle,
                            )
                            row_data[f"{metric}_SE"] = se[rna_type]
                    else:
                        row_data[f"{metric}_Mean"] = np.nan
                        if calculate_se:
                            row_data[f"{metric}_SE"] = np.nan

                results.append(row_data)

            # Then calculate "All RNA types" average for this depth
            row_data = {"Model": model, "Depth": depth, "RNA_TYPE": "All"}

            for metric in ["Spearman", "AUC", "MCC"]:
                # Calculate the mean of type means for this depth
                type_means = []
                for rna_type in types:
                    type_datasets = wt_seqs[wt_seqs["RNA_TYPE"] == rna_type][
                        "DMS_ID"
                    ].tolist()
                    type_subset = wt_seqs[wt_seqs["DMS_ID"].isin(type_datasets)]

                    col_name = f"{metric}_{model}_{depth}"
                    values = type_subset[col_name].dropna()
                    if len(values) > 0:
                        type_means.append(np.mean(values))

                if type_means:
                    # 'All' is average of type means
                    row_data[f"{metric}_Mean"] = np.mean(type_means)

                    if calculate_se:
                        # For "All", combine SEs from individual types
                        ses = []
                        for rna_type in types:
                            type_datasets = wt_seqs[wt_seqs["RNA_TYPE"] == rna_type][
                                "DMS_ID"
                            ].tolist()
                            if type_datasets:
                                type_subset = wt_seqs[
                                    wt_seqs["DMS_ID"].isin(type_datasets)
                                ]
                                col_name = f"{metric}_{model}_{depth}"
                                metric_columns = [
                                    f"{metric}_{col}_{depth}" for col in score_columns
                                ]

                                se = bootstrap_se(
                                    type_subset,
                                    "DMS_ID",
                                    col_name,
                                    [rna_type],
                                    metric_columns,
                                    number_assay_reshuffle,
                                )
                                if not np.isnan(se[rna_type]):
                                    ses.append(se[rna_type] ** 2)  # Variance

                        if ses:
                            # Standard error is sqrt of average variance
                            row_data[f"{metric}_SE"] = np.sqrt(np.mean(ses))
                        else:
                            row_data[f"{metric}_SE"] = np.nan
                else:
                    row_data[f"{metric}_Mean"] = np.nan
                    if calculate_se:
                        row_data[f"{metric}_SE"] = np.nan

            results.append(row_data)

    return pd.DataFrame(results)


def save_assay_level_results_transposed(
    wt_seqs: pd.DataFrame, score_columns: List[str], output_file: str
):
    """Save assay-level results grouped by metric type."""
    output_columns = ["DMS_ID", "RNA_TYPE"]

    # Outer loop: Metric, Inner loop: Model -> Groups by Metric
    for metric in ["Spearman", "AUC", "MCC"]:
        for model in score_columns:
            output_columns.append(f"{metric}_{model}")

    final_columns = [col for col in output_columns if col in wt_seqs.columns]
    wt_seqs.to_csv(output_file, columns=final_columns, index=False)


def main(args):
    wt_seqs = pd.read_csv(args.reference_file)
    model_list = [
        "evo1",
        "evo1.5",
        "evo2",
        "GenSLM",
        "NT",
        "rinalmo",
        "RNAErnie",
        "RNA-FM",
    ]
    score_columns = [model + str("_score") for model in model_list]
    wt_seqs = analyze_datasets(wt_seqs, args.combined_dir, score_columns)
    types = ["mRNA-splicing", "mRNA-coding", "tRNA", "Aptamer", "Ribozyme"]

    # Calculate averages and standard errors per RNA type
    result_df_type = calculate_RNA_types_averages_with_se(
        wt_seqs, types, score_columns, args.calculate_se
    )

    # Calculate averages and standard errors per mutation depth
    result_df_mutation_depth = calculate_mutation_depth_averages_with_se(
        wt_seqs, score_columns, args.combined_dir, False
    )

    # Calculate combined averages (RNA type + mutation depth)
    result_df_combined = calculate_combined_averages_with_se(
        wt_seqs, types, score_columns, args.combined_dir, False
    )

    # Ensure the performance directory exists
    os.makedirs(args.performance_dir, exist_ok=True)

    # Save results to CSV
    result_df_type.to_csv(
        os.path.join(args.performance_dir, "results_by_rna_type.csv"), index=False
    )
    result_df_mutation_depth.to_csv(
        os.path.join(args.performance_dir, "results_by_mutation_depth.csv"), index=False
    )
    result_df_combined.to_csv(
        os.path.join(args.performance_dir, "results_by_rna_type_and_depth.csv"),
        index=False,
    )

    # Save assay-level results
    save_assay_level_results(
        wt_seqs,
        score_columns,
        os.path.join(args.performance_dir, "assay_level_results.csv"),
    )
    save_assay_level_results_transposed(
        wt_seqs,
        score_columns,
        os.path.join(args.performance_dir, "assay_level_results_transposed.csv"),
    )

    # Print results
    print("\nMetrics by RNA Type:")
    print(result_df_type)
    print("\nMetrics by Mutation Depth:")
    print(result_df_mutation_depth)
    print("\nMetrics by RNA Type and Mutation Depth:")
    print(result_df_combined)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze RNA datasets and calculate metrics for multiple models."
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default="reference_sheet_cleaned_CAS.csv",
        help="Path to the reference CSV file",
    )
    parser.add_argument(
        "--combined_dir",
        type=str,
        default="combined_results",
        help="Base directory for combined result files",
    )
    parser.add_argument(
        "--calculate_se",
        action="store_true",
        help="Calculate standard errors (computationally intensive)",
    )
    parser.add_argument(
        "--performance_dir",
        type=str,
        default=None,
        help="Dir where performance file should be stored",
    )
    args = parser.parse_args()
    if args.performance_dir is None:
        args.performance_dir = os.getcwd()
    main(args)
