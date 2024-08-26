import argparse
import os
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from arnie.bpps import bpps
from utils import (load_data, unpaired_probabilities, process_predictions, 
                   compute_performance_metrics, save_predictions, save_performance_metrics)

def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RNA Structure Prediction using Arnie")
    parser.add_argument("model_type", type=str, help="Arnie model type to use for prediction")
    parser.add_argument("--data_folder", type=str, default="./data", help="Path to data folder")
    parser.add_argument("--output_folder", type=str, default="./output", help="Path to output folder")
    parser.add_argument("--test_data_name", type=str, default="final_test_set.csv", help="Name of test data file")
    parser.add_argument("--performance_file", type=str, default="performance_SS_pred.csv", help="Name of performance summary file")
    return parser.parse_args()

def predict_structures_arnie(seq: str, model_type: str) -> np.ndarray:
    pred = bpps(seq, package=model_type)
    return unpaired_probabilities(pred)

def main(args: argparse.Namespace):
    # Setup paths
    test_data_path = os.path.join(args.data_folder, 'test_data', args.test_data_name)
    output_path = os.path.join(args.output_folder, f'predictions_{args.model_type}.csv')
    
    # Load data
    test_data = load_data(test_data_path)
    
    # Check if predictions already exist
    if os.path.exists(output_path):
        predictions = load_data(output_path)
    else:
        unique_sequences = test_data[["sequence_id", "sequence"]].drop_duplicates()
        predictions = []
        for _, row in tqdm(unique_sequences.iterrows(), total=len(unique_sequences)):
            pred = predict_structures_arnie(row['sequence'], args.model_type)
            pred_df = pd.DataFrame({
                'sequence': [row['sequence']] * len(pred),
                'sequence_id': [row['sequence_id']] * len(pred),
                'position_id': range(len(pred)),
                f'prediction_{args.model_type}': pred
            })
            predictions.append(pred_df)
        predictions = pd.concat(predictions)
        
        # Save predictions
        save_predictions(predictions, output_path)
    
    # Process predictions and compute performance metrics
    processed_data = process_predictions(predictions, test_data, args.model_type)
    metrics = compute_performance_metrics(processed_data, args.model_type)
    
    # Save and print performance metrics
    save_performance_metrics(metrics, args.performance_file)

if __name__ == "__main__":
    args = setup_argparse()
    main(args)