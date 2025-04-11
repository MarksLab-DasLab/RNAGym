import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error
from typing import Tuple, List, Dict

def array_to_string(arr, precision=4):
    x = np.array2string(arr, separator=",", precision=precision)
    x = x.replace("\n", "").replace(" ", "")
    return x

def string_to_array(arr_str):
    x = np.fromstring(arr_str[1:-1], dtype=float, sep=",")
    return x

def load_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        return pd.read_csv(file_path, low_memory=False)

def unpaired_probabilities(prob_matrix: np.ndarray) -> np.ndarray:
    return np.prod(1 - prob_matrix, axis=1)

def compute_bins(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    for col in ['reactivity_DMS_MaP', f'prediction_{model_type}']:
        median = df[col].median()
        df[f'{col}_bin'] = df[col] > median
    return df

def process_predictions(predictions: pd.DataFrame, test_data: pd.DataFrame, model_type: str) -> pd.DataFrame:
    merged_data = pd.merge(test_data, predictions, on=['seqID', "position_id"], how="left")
    merged_data = merged_data[merged_data[f'prediction_{model_type}'].notna()]
    #merged_data = merged_data[merged_data["Usage"] != "Ignored"]
    
    for col in ['reactivity_DMS_MaP', f'prediction_{model_type}']:
        merged_data[col] = merged_data[col].clip(0, 1)
    
    return merged_data.groupby('sequence_id', group_keys=False).apply(compute_bins, model_type=model_type)

def collate_data(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """ Compiles the true and predicted reactivity into one long np array """
    true_reactivities = []
    pred_probs = []
    for reactivity, pred in df.values:
        reactivity = string_to_array(reactivity).clip(0, 1)
        pred = string_to_array(pred).clip(0, 1)
        assert(len(pred) == len(reactivity))
        
        true_reactivities.append(reactivity)
        pred_probs.append(pred)
    true_reactivities = np.hstack(true_reactivities)
    pred_probs = np.hstack(pred_probs)
    
    # remove nans
    mask = ~np.isnan(true_reactivities)
    true_reactivities = true_reactivities[mask]
    pred_probs = pred_probs[mask]
    
    # compute binned values
    true_values = true_reactivities > np.median(true_reactivities)
    pred_values = pred_probs > np.median(pred_probs)
    return true_reactivities, pred_probs, true_values, pred_values

def compute_performance_metrics(predictions: pd.DataFrame, model_type: str) -> List[dict]:
    metrics = []
    for chemical_modifier in ["DMS", "2A3"]:
        reactivity_col = f"reactivity_{chemical_modifier}"
        df_sub = predictions[[reactivity_col, f"prediction_{model_type}"]]
        
        # drop rows with missing reactivity data or missing prediction
        df_sub = df_sub.dropna()

        true_reactivities, pred_probs, true_values, pred_values = collate_data(df_sub)
        metric = {
            'Chemical Modifier': chemical_modifier,
            'Model Type': model_type,
            'AUC': roc_auc_score(true_values, pred_probs),
            'MCC': matthews_corrcoef(true_values, pred_values),
            'MAE': mean_absolute_error(true_reactivities, pred_probs),
            'F1-Score': f1_score(true_values, pred_values, average='macro'),
            'Total sequence length': len(true_reactivities),
            }
        metrics.append(metric)
    return metrics

def save_predictions(predictions: pd.DataFrame, output_path: str):
    predictions.to_csv(output_path, index=False)

def save_performance_metrics(metrics: List[Dict], performance_file: str):
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    metrics_df.to_csv(performance_file, mode='a', index=False, header=not os.path.exists(performance_file))