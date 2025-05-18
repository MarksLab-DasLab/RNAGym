#!/usr/bin/env python3
"""
MXfold2 Scoring Script

This script processes MXfold2 outputs and converts them to a 
format compatible with RNA-FM for direct comparison.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import subprocess
import tempfile
import re
from Bio import SeqIO
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run MXfold2 and process outputs for scoring")
    
    parser.add_argument("--fasta", required=True, help="Path to FASTA file with RNA sequences")
    parser.add_argument("--output_dir", required=True, help="Directory to store MXfold2 output files")
    parser.add_argument("--output_file", default=None, help="Output CSV file (default: {output_dir}/mxfold2_scores.csv)")
    parser.add_argument("--reference_data", default=None, help="Path to reference data CSV file (e.g., DMS reactivity data)")
    parser.add_argument("--performance_file", default=None, help="Path to save performance metrics")
    parser.add_argument("--param", default="", help="MXfold2 parameter file (default: use built-in parameters)")
    parser.add_argument("--model", default="Turner", 
                        choices=["Turner", "Zuker", "ZukerS", "ZukerL", "ZukerC", "Mix", "MixC"],
                        help="MXfold2 model type (default: Turner)")
    parser.add_argument("--skip_prediction", action="store_true", help="Skip running MXfold2 prediction (use existing outputs)")
    parser.add_argument("--clean_ids", action="store_true", help="Clean sequence IDs (replace '.' with '_')")
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = os.path.join(args.output_dir, "mxfold2_scores.csv")
    
    return args

def read_sequences(fasta_file, clean_ids=False):
    """Read sequences from a FASTA file"""
    sequences = {}
    id_mapping = {}
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        original_id = record.id
        
        if clean_ids:
            # Replace periods and special characters with underscores
            clean_id = re.sub(r'[.+]', '_', original_id)
            # Store mapping for later retrieval
            id_mapping[clean_id] = original_id
            sequences[clean_id] = str(record.seq).upper()
        else:
            sequences[original_id] = str(record.seq).upper()
    
    return sequences, id_mapping

def clean_filename(filename):
    """Clean a filename by replacing periods and special characters with underscores"""
    return re.sub(r'[.+]', '_', filename)

def run_mxfold2(fasta_file, output_dir, param="", model="Turner", clean_ids=False):
    """Run MXfold2 on the input FASTA file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual FASTA files for each sequence
    sequences, id_mapping = read_sequences(fasta_file, clean_ids)
    fasta_files = []
    
    # Create a temporary directory for the individual FASTA files
    temp_dir = os.path.join(output_dir, "temp_fasta")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Use a counter for short filenames
    for i, (seq_id, sequence) in enumerate(sequences.items()):
        seq_fasta = os.path.join(temp_dir, f"seq_{i}.fa")
        
        with open(seq_fasta, "w") as f:
            f.write(f">{seq_id}\n{sequence}\n")
        fasta_files.append((seq_id, seq_fasta))
    
    # Run MXfold2 on each sequence
    for seq_id, seq_fasta in tqdm(fasta_files, desc="Running MXfold2"):
        # Construct command
        cmd = ["mxfold2", "predict"]
        if param:
            cmd.extend(["--param", param])
        cmd.extend(["--model", model])
        cmd.extend(["--bpp", output_dir])
        cmd.extend(["--bpseq", output_dir])
        cmd.append(seq_fasta)
        
        # Run MXfold2
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error running MXfold2 on {seq_id}: {e.stderr.decode()}")
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    return id_mapping

def find_matching_file(base_name, file_list):
    """Find a file in the list that matches the base name"""
    # First, try exact match
    for file_path in file_list:
        if os.path.basename(file_path).startswith(f"{base_name}."):
            return file_path
            
    # If no exact match, try stripping after first period
    base_name_short = base_name.split('.')[0]
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        if file_name.startswith(f"{base_name_short}."):
            return file_path
    
    # If still no match, try cleaned version
    clean_base = clean_filename(base_name)
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        if file_name.startswith(f"{clean_base}."):
            return file_path
            
    return None

def process_bpp_file(bpp_file, sequence_length):
    """Process MXfold2 base-pairing probability matrix file"""
    try:
        # Load probability matrix
        bpp_matrix = np.loadtxt(bpp_file)
        
        # Ensure matrix is the right shape
        if bpp_matrix.shape != (sequence_length, sequence_length):
            print(f"Warning: Matrix shape {bpp_matrix.shape} does not match sequence length {sequence_length}")
            if min(bpp_matrix.shape) >= sequence_length:
                # Truncate matrix if it's bigger than sequence
                bpp_matrix = bpp_matrix[:sequence_length, :sequence_length]
            else:
                # Can't use this matrix if it's too small
                return None
        
        return bpp_matrix
    except Exception as e:
        print(f"Error processing {bpp_file}: {e}")
        return None

def unpaired_probabilities(bpp_matrix):
    """
    Calculate probability of each position being unpaired:
    P(unpaired) = 1 - sum(P(paired with any other base))
    """
    # Sum across rows to get total pairing probability for each position
    paired_probs = np.sum(bpp_matrix, axis=1)
    # Ensure probabilities are bounded between 0 and 1
    paired_probs = np.clip(paired_probs, 0, 1)
    # Calculate unpaired probabilities
    return 1 - paired_probs

def main():
    """Main function"""
    args = parse_arguments()
    
    # Check if input files exist
    if not os.path.exists(args.fasta):
        print(f"Error: FASTA file {args.fasta} not found")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read sequences
    print(f"Reading sequences from {args.fasta}")
    sequences, _ = read_sequences(args.fasta, args.clean_ids)
    print(f"Read {len(sequences)} sequences")
    
    # Store the mapping of original to clean IDs
    id_mapping = {}
    
    # Run MXfold2 if not skipped
    if not args.skip_prediction:
        print(f"Running MXfold2 on {len(sequences)} sequences")
        id_mapping = run_mxfold2(args.fasta, args.output_dir, args.param, args.model, args.clean_ids)
    
    # Find all MXfold2 output files
    bpp_files = glob.glob(os.path.join(args.output_dir, "*.bpp"))
    print(f"Found {len(bpp_files)} MXfold2 base-pairing probability files")
    
    if len(bpp_files) == 0:
        print("Error: No MXfold2 output files found. Make sure MXfold2 has been run on your sequences.")
        sys.exit(1)
    
    # Process each sequence
    results = []
    
    seq_ids_processed = set()
    
    for seq_id, sequence in tqdm(sequences.items(), desc="Processing MXfold2 outputs"):
        # Find the corresponding BPP file
        bpp_file = find_matching_file(seq_id, bpp_files)
        
        if bpp_file is None:
            print(f"Warning: No BPP file found for sequence {seq_id}, skipping...")
            continue
        
        seq_length = len(sequence)
        
        # Process probability matrix
        bpp_matrix = process_bpp_file(bpp_file, seq_length)
        if bpp_matrix is None:
            continue
        
        # Calculate unpaired probabilities
        unpaired_probs = unpaired_probabilities(bpp_matrix)
        
        # Use original ID if we have a mapping
        original_id = id_mapping.get(seq_id, seq_id)
        
        # Add results for this sequence
        for i in range(seq_length):
            results.append({
                'sequence_id': original_id,
                'sequence': sequence,
                'position_id': i,
                'prediction_MXfold2': unpaired_probs[i]
            })
        
        seq_ids_processed.add(seq_id)
    
    print(f"Successfully processed {len(seq_ids_processed)} sequences out of {len(sequences)}")
    
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
            processed_df = process_predictions(predictions_df, reference_df, model_type="MXfold2")
            metrics = compute_performance_metrics(processed_df, model_type="MXfold2")
            
            print("\nPerformance Metrics:")
            for metric, value in metrics.items():
                if metric != 'Model Type':
                    print(f"{metric}: {value:.4f}")
            
            # Save metrics if output file specified
            if args.performance_file:
                save_performance_metrics(metrics, args.performance_file)
                print(f"Performance metrics saved to {args.performance_file}")
    else:
        print("No results to save")

def load_data(file_path):
    """Load data from a CSV file"""
    return pd.read_csv(file_path, low_memory=False)

def compute_bins(df, model_type):
    """Compute binary classifications based on median values"""
    for col in ['reactivity_DMS_MaP', f'prediction_{model_type}']:
        if col in df.columns:
            median = df[col].median()
            df[f'{col}_bin'] = df[col] > median
    return df

def process_predictions(predictions, test_data, model_type):
    """Process predictions and merge with test data"""
    merged_data = pd.merge(test_data, predictions, on=['sequence_id', 'position_id'], how='left')
    merged_data = merged_data[merged_data[f'prediction_{model_type}'].notna()]
    return merged_data.groupby('sequence_id', group_keys=False).apply(compute_bins, model_type=model_type)

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