import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, low_memory=False)

def unpaired_probabilities(prob_matrix: np.ndarray) -> np.ndarray:
    return np.prod(1 - prob_matrix, axis=1)

def compute_bins(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    for col in ['reactivity_DMS_MaP', f'prediction_{model_type}']:
        median = df[col].median()
        df[f'{col}_bin'] = df[col] > median
    return df

def process_predictions(predictions: pd.DataFrame, test_data: pd.DataFrame, model_type: str) -> pd.DataFrame:
    merged_data = pd.merge(test_data, predictions, on=['sequence_id', "position_id"], how="left")
    merged_data = merged_data[merged_data[f'prediction_{model_type}'].notna()]
    merged_data = merged_data[merged_data["Usage"] != "Ignored"]
    
    for col in ['reactivity_DMS_MaP', f'prediction_{model_type}']:
        merged_data[col] = merged_data[col].clip(0, 1)
    
    return merged_data.groupby('sequence_id', group_keys=False).apply(compute_bins, model_type=model_type)

def compute_performance_metrics(df: pd.DataFrame, model_type: str) -> dict:
    true_values = df['reactivity_DMS_MaP_bin']
    pred_values = df[f'prediction_{model_type}_bin']
    pred_probs = df[f'prediction_{model_type}']
    true_reactivities = df['reactivity_DMS_MaP']
    
    return {
        'Model Type': model_type,
        'AUC': roc_auc_score(true_values, pred_probs),
        'MCC': matthews_corrcoef(true_values, pred_values),
        'MAE': mean_absolute_error(true_reactivities, pred_probs),
        'F1-Score': f1_score(true_values, pred_values, average='macro')
    }

def save_predictions(predictions: pd.DataFrame, output_path: str):
    predictions.to_csv(output_path, index=False)

def save_performance_metrics(metrics: dict, performance_file: str):
    metrics_df = pd.DataFrame([metrics])
    print(metrics_df)
    metrics_df.to_csv(performance_file, mode='a', index=False, header=not os.path.exists(performance_file))