import argparse
import os
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import numpy as np
from arnie.bpps import bpps
from utils import (load_data, unpaired_probabilities, process_predictions, 
                   compute_performance_metrics, save_predictions, save_performance_metrics,
                   array_to_string, string_to_array)

def organize_data(df: pd.DataFrame) -> pd.DataFrame:
    # Organize the DMS and 2A3 data
    df_DMS = df[df["modifier"] == "DMS"]
    df_2A3 = df[df["modifier"] == "2A3"]

    for df_mod in [df_DMS, df_2A3]:
        # Remove duplicate sequences within each df. Select the duplicate with the highest signal-to-noise ratio (SNR)
        df_mod = df_mod.sort_values(['sequence', 'SNR'], ascending=[False, False])
        df_mod = df_mod.drop_duplicates(subset=['sequence'], keep="first")
    df = pd.merge(df_DMS, df_2A3, on='sequence', how='outer', suffixes=('_DMS', '_2A3'))
    df = df[["seqID_DMS", "seqID_2A3", "sequence", "reactivity_DMS", "reactivity_2A3"]]
    return df

def predict_structures_arnie(seq: str, model_type: str) -> np.ndarray:
    # clean sequence
    seq = seq.replace("T", "U") # Convert to RNA if there's any T's by accident
    pred = bpps(seq, package=model_type)
    return unpaired_probabilities(pred)

def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RNA Structure Prediction using Arnie")
    parser.add_argument("model_type", type=str, help="Arnie model type to use for prediction")
    parser.add_argument("--data_folder", type=str, default="./data", help="Path to data folder")
    parser.add_argument("--output_folder", type=str, default="./output", help="Path to output folder")
    parser.add_argument("--test_data_name", type=str, default="final_test_set.csv", help="Name of test data file")
    parser.add_argument("--performance_file", type=str, default="performance_SS_pred.csv", help="Name of performance summary file")
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Setup paths
    test_data_path = os.path.join(args.data_folder, 'test_data', args.test_data_name)
    output_path = os.path.join(args.output_folder, f'predictions_{args.model_type}.csv')
    os.makedirs(args.output_folder, exist_ok=True)

    # Load data
    df = load_data(test_data_path)

    # Organize and clean up such that each row is a unique sequence
    df = organize_data(df)

    # Check if predictions already exist
    if os.path.exists(output_path):
        predictions = load_data(output_path)
    else:
        # Append to the last column the model predictions
        df[f"prediction_{args.model_type}"] = pd.Series(dtype='object')
        for rowidx, row in tqdm(df.iterrows(), total=df.shape[0]):
            pred = predict_structures_arnie(row['sequence'], args.model_type)
            df.at[rowidx, f"prediction_{args.model_type}"] = array_to_string(pred)

        # Save predictions
        save_predictions(df, output_path)

    # Process predictions and compute performance metrics
    metrics = compute_performance_metrics(processed_data, args.model_type)
    
    # Save and print performance metrics
    save_performance_metrics(metrics, args.performance_file)

if __name__ == "__main__":
    args = setup_argparse()
    main(args)