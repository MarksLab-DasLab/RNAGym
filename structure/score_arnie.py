import argparse
import os
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import numpy as np
from arnie.bpps import bpps
from utils import (load_data, unpaired_probabilities, process_predictions, organize_data,
                   compute_performance_metrics, save_predictions, save_performance_metrics,
                   array_to_string, string_to_array)
from tqdm.contrib.concurrent import process_map
from functools import partial

def predict_structures_arnie(seq: str, model_type: str) -> np.ndarray:
    # clean sequence
    seq = seq.replace("T", "U") # Convert to RNA if there's any T's by accident
    pred = bpps(seq, package=model_type)
    return unpaired_probabilities(pred)

def predict_sequence(seq: str, model_type: str) -> str:
    pred = predict_structures_arnie(seq, model_type)
    pred = array_to_string(pred)
    return pred

def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RNA Structure Prediction using Arnie")
    parser.add_argument("--model_type", type=str, help="Arnie model type to use for prediction")
    parser.add_argument("--data_folder", type=str, default="./data", help="Path to data folder")
    parser.add_argument("--output_folder", type=str, default="./output", help="Path to output folder")
    parser.add_argument("--test_data_name", type=str, default="final_test_set.csv", help="Name of test data file")
    parser.add_argument("--performance_file", type=str, default="performance_SS_pred.csv", help="Name of performance summary file")
    parser.add_argument("--num_workers", type=int, default=0, help="run multiprocessing if num_workers > 0.")
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
        df = load_data(output_path)
    else:
        func = partial(predict_sequence, model_type=args.model_type)
        if args.num_workers == 0:
            # Append to the last column the model predictions
            df[f"prediction_{args.model_type}"] = pd.Series(dtype='object')
            for rowidx, row in tqdm(df.iterrows(), total=df.shape[0]):
                df.at[rowidx, f"prediction_{args.model_type}"] = func(row['sequence'])
        else:
            results = process_map(func, df.sequence.values, max_workers=args.num_workers, chunksize=1)
            df[f"prediction_{args.model_type}"] = results

        # Save predictions
        save_predictions(df, output_path)

    # Process predictions and compute performance metrics
    metrics = compute_performance_metrics(df, args.model_type)
    
    # Save and print performance metrics
    save_performance_metrics(metrics, args.performance_file)

if __name__ == "__main__":
    args = setup_argparse()
    main(args)