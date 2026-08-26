#!/usr/bin/env python3
"""
UFold RNA Unpaired Probability Calculator

This script calculates UFold-style unpaired probabilities for RNA sequences and 
outputs them in a format compatible with RNA-FM for direct comparison.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error

from ..utils import (load_data, unpaired_probabilities, organize_data,
                   compute_performance_metrics, save_predictions, save_performance_metrics)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Calculate UFold-style unpaired probabilities for RNA sequences")
    
    parser.add_argument("--fasta", required=True, help="Path to FASTA file with RNA sequences")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--output_file", default=None, help="Output CSV file (default: {output_dir}/ufold_scores.csv)")
    parser.add_argument("--reference_data", default=None, help="Path to reference data CSV file (e.g., DMS reactivity data)")
    parser.add_argument("--performance_file", default=None, help="Path to save performance metrics")
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = os.path.join(args.output_dir, "ufold_scores.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def read_sequences(fasta_file):
    """Read sequences from a FASTA file"""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq).upper().replace('T', 'U')
    return sequences

def generate_pair_matrix(sequence):
    """Generate pair probability matrix using UFold's base-pairing rules"""
    seq_len = len(sequence)
    pair_matrix = np.zeros((seq_len, seq_len))
    
    # Define base-pairing scores (similar to UFold's constraints)
    pair_scores = {
        ('A', 'U'): 2.0,
        ('U', 'A'): 2.0,
        ('G', 'C'): 3.0,
        ('C', 'G'): 3.0,
        ('G', 'U'): 0.8,
        ('U', 'G'): 0.8
    }
    
    # Calculate initial pairing scores
    for i in range(seq_len):
        for j in range(i+4, seq_len):  # Minimum loop size of 3
            pair = (sequence[i], sequence[j])
            if pair in pair_scores:
                # Get base pairing score
                score = pair_scores[pair]
                
                # Apply distance penalty (similar to UFold's approach)
                distance_factor = np.exp(-0.005 * (j - i))
                
                # Check for neighboring pairs to enhance stability
                neighbor_bonus = 1.0
                if i > 0 and j < seq_len-1:
                    if (sequence[i-1], sequence[j+1]) in pair_scores:
                        neighbor_bonus = 1.2
                if i < seq_len-1 and j > 0:
                    if (sequence[i+1], sequence[j-1]) in pair_scores:
                        neighbor_bonus = 1.2
                
                # Calculate final pair probability
                pair_prob = min(0.95, score * distance_factor * neighbor_bonus / 3.0)
                
                # Store in matrix
                pair_matrix[i, j] = pair_prob
                pair_matrix[j, i] = pair_prob  # Make matrix symmetric
    
    return pair_matrix

def unpaired_probabilities(pair_matrix):
    """
    Calculate probability of each position being unpaired using RNA-FM's method:
    P(unpaired) = product(1 - P(paired with any other base))
    """
    return np.prod(1 - pair_matrix, axis=1)

def main():
    """Main function"""
    args = parse_arguments()


    if os.path.exists( args.output_file):
        print("Loading predictions from disk")
        predictions = load_data(args.output_path)

    else:
    
        # Check if input file exists
        if not os.path.exists(args.fasta):
            print(f"Error: FASTA file {args.fasta} not found")
            sys.exit(1)


        
        # Read sequences
        print(f"Reading sequences from {args.fasta}")
        sequences = read_sequences(args.fasta)
        print(f"Read {len(sequences)} sequences")
        
        # Process each sequence
        results = []
        
        for seq_id, sequence in tqdm(sequences.items(), desc="Processing sequences"):
            # Generate pair probability matrix
            pair_matrix = generate_pair_matrix(sequence)
            
            # Calculate unpaired probabilities
            unpaired_probs = unpaired_probabilities(pair_matrix)
            
            # Add results for this sequence
            for i in range(len(sequence)):
                results.append({
                    'sequence_id': seq_id,
                    'sequence': sequence,
                    'position_id': i,
                    'prediction_UFold': unpaired_probs[i]
                })
        
        # Create DataFrame and save to CSV
        if results:
            predictions_df = pd.DataFrame(results)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
            
            # Save to CSV
            predictions_df.to_csv(args.output_file, index=False)
            print(f"Results saved to {args.output_file}")
            
            # Print summary
            num_sequences = predictions_df['sequence_id'].nunique()
            avg_seq_length = predictions_df.groupby('sequence_id').size().mean()
            
            print("\nSummary:")
            print(f"Processed {num_sequences} sequences")
            print(f"Average sequence length: {avg_seq_length:.1f}")
            
    # Compute performance metrics if reference data is provided
    if args.reference_data and os.path.exists(args.reference_data):
        print(f"\nComputing performance metrics using reference data: {args.reference_data}")
        reference_df = load_data(args.reference_data)
        
        # Process and compute metrics
        processed_df = process_predictions(predictions_df, reference_df, model_type="UFold")
        metrics = compute_performance_metrics(processed_df, model_type="UFold")
        
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            if metric != 'Model Type':
                print(f"{metric}: {value:.4f}")
        
        # Save metrics if output file specified
        if args.performance_file:
            save_performance_metrics(metrics, args.performance_file)
 

def load_data(file_path):
    """Load data from a CSV file"""
    return pd.read_parquet(file_path, low_memory=False)

def compute_bins(df, model_type):
    """Compute binary classifications based on median values"""
    for col in ['reactivity_DMS_MaP', f'prediction_{model_type}']:
        if col in df.columns:
            median = df[col].median()
            df[f'{col}_bin'] = df[col] > median
    return df

def process_predictions(predictions, test_data, model_type):
    """Process predictions and merge with test data"""
    merged_data = pd.merge(test_data, predictions, on=['sequence'], how='left')
    merged_data = merged_data[merged_data[f'prediction_{model_type}'].notna()]
    return merged_data.groupby('seqID', group_keys=False).apply(compute_bins, model_type=model_type)

def compute_performance_metrics(df, model_type):
    """Compute performance metrics for model predictions"""
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

def save_performance_metrics(metrics, performance_file):
    """Save performance metrics to a CSV file"""
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(performance_file, mode='a', index=False, header=not os.path.exists(performance_file))

if __name__ == "__main__":
    main()