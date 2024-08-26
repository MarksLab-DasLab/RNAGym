import argparse
import os
from typing import List, Dict
import pandas as pd
import torch
import torch.nn.functional as F
import fm
from tqdm import tqdm
from utils import (load_data, unpaired_probabilities, process_predictions, 
                   compute_performance_metrics, save_predictions, save_performance_metrics)

def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RNA Structure Prediction using RNA-FM")
    parser.add_argument("--data_folder", type=str, default="./data", help="Path to data folder")
    parser.add_argument("--output_folder", type=str, default="./output", help="Path to output folder")
    parser.add_argument("--test_data_name", type=str, default="final_test_set.csv", help="Name of test data file")
    parser.add_argument("--performance_file", type=str, default="performance_SS_pred.csv", help="Name of performance summary file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for RNA-FM model")
    return parser.parse_args()

def setup_rna_fm_model():
    model, alphabet = fm.downstream.build_rnafm_resnet(type="ss")
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"RNA-FM model is on device: {device}")
    return model, batch_converter, device

def score_rna_fm(model, batch_converter, device, batch_data: List[Dict[str, str]]) -> torch.Tensor:
    batch_data = [tuple(data.values()) for data in batch_data]
    _, _, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model({"token": batch_tokens})
    
    logits = results["r-ss"]
    probabilities = F.softmax(logits, dim=-1)
    return probabilities

def main(args: argparse.Namespace):
    model_type = "RNA_FM"
    # Setup paths
    test_data_path = os.path.join(args.data_folder, 'test_data', args.test_data_name)
    output_path = os.path.join(args.output_folder, f'predictions_{model_type}.csv')
    
    # Load data
    test_data = load_data(test_data_path)
    
    # Check if predictions already exist
    if os.path.exists(output_path):
        predictions = load_data(output_path)
    else:
        unique_sequences = test_data[["sequence_id", "sequence"]].drop_duplicates()
        model, batch_converter, device = setup_rna_fm_model()
        all_predictions = []
        
        for i in tqdm(range(0, len(unique_sequences), args.batch_size)):
            batch_data = unique_sequences.iloc[i:i + args.batch_size].to_dict('records')
            batch_predictions = score_rna_fm(model, batch_converter, device, batch_data)
            
            for j, row in enumerate(batch_data):
                pred = unpaired_probabilities(batch_predictions[j].cpu().numpy())
                pred_df = pd.DataFrame({
                    'sequence': [row['sequence']] * len(pred),
                    'sequence_id': [row['sequence_id']] * len(pred),
                    'position_id': range(len(pred)),
                    f'prediction_{model_type}': pred
                })
                all_predictions.append(pred_df)
        
        predictions = pd.concat(all_predictions)
        
        # Save predictions
        save_predictions(predictions, output_path)
    
    # Process predictions and compute performance metrics
    processed_data = process_predictions(predictions, test_data, model_type)
    metrics = compute_performance_metrics(processed_data, model_type)
    
    # Save and print performance metrics
    save_performance_metrics(metrics, args.performance_file)

if __name__ == "__main__":
    args = setup_argparse()
    main(args)