#!/usr/bin/env python3
"""
Script to run Evo model inference on DMS assay sequences.
Takes a reference sheet and row ID to process specific assays.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from evo import Evo
from evo.scoring import prepare_batch, logits_to_logprobs
from torch.utils.data import Dataset, DataLoader


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess RNA sequence for DNA model:
    - Convert RNA (U) to DNA (T)
    - Convert to uppercase
    - Remove any whitespace

    Args:
        sequence: Input RNA or DNA sequence

    Returns:
        str: Preprocessed DNA sequence
    """
    return sequence.strip().upper().replace("U", "T")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Evo model inference on DMS assay sequences."
    )
    parser.add_argument(
        "--row_id",
        type=int,
        required=True,
        help="Row ID in the reference sheet to process",
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=True,
        help="Path to reference sheet containing DMS_ID column",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=True,
        help="Directory containing DMS CSV files",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="evo-1-8k-base",
        help="Name of the Evo model to use (default: evo-1-8k-base)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for processing sequences (default: 128)",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """
    Load reference sheet and get DMS_ID for specified row.

    Args:
        ref_sheet_path: Path to reference sheet
        row_id: Row ID to process

    Returns:
        str: DMS_ID for specified row

    Raises:
        ValueError: If row_id is not found or DMS_ID is missing
    """
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        if row_id >= len(ref_df):
            raise ValueError(
                f"Row ID {row_id} exceeds number of rows in reference sheet"
            )

        dms_id = ref_df.loc[row_id, "DMS_ID"]
        if pd.isna(dms_id):
            raise ValueError(f"DMS_ID is missing for row {row_id}")

        return str(dms_id)

    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found: {ref_sheet_path}")
    except KeyError:
        raise KeyError("Reference sheet must contain 'DMS_ID' column")


def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """
    Load DMS data for specified DMS_ID.

    Args:
        dms_dir_path: Directory containing DMS files
        dms_id: DMS ID to process

    Returns:
        pd.DataFrame: DataFrame containing sequences to process

    Raises:
        FileNotFoundError: If DMS file is not found
    """
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")

    df = pd.read_csv(dms_file)
    required_cols = ["mutant", "DMS_score", "sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DMS file: {missing_cols}")

    return df


class SequenceDataset(Dataset):
    """Dataset for processing sequences in batches."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def run_inference(
    model, tokenizer, sequences: list, device: str, batch_size: int
) -> np.ndarray:
    """
    Run Evo model inference on sequences in batches.

    Args:
        model: Loaded Evo model
        tokenizer: Evo tokenizer
        sequences: List of sequences to process
        device: Device to run inference on
        batch_size: Number of sequences to process in each batch

    Returns:
        np.ndarray: Log probabilities array
    """
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logprobs = []

    # Create progress bar
    total_batches = len(dataloader)
    pbar = tqdm(total=len(sequences), desc="Scoring sequences", unit="seq")

    for batch_seqs in dataloader:
        input_ids, seq_lengths = prepare_batch(
            batch_seqs,
            tokenizer,
            prepend_bos=True,
            device=device,
        )
        assert len(seq_lengths) == input_ids.shape[0]

        with torch.inference_mode():
            logits, *_ = model(input_ids)

        # Calculate log probabilities
        logprobs = logits_to_logprobs(logits, input_ids, trim_bos=True)

        # Store batch results
        all_logprobs.append(logprobs.float().cpu().numpy())

        # Update progress bar by batch size
        pbar.update(len(batch_seqs))

    pbar.close()

    # Concatenate all batches
    return np.concatenate(all_logprobs)


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load DMS ID from reference sheet
        dms_id = load_reference_data(args.ref_sheet, args.row_id)
        print(f"Processing DMS ID: {dms_id}")

        # Load DMS data
        dms_df = load_dms_data(args.dms_dir_path, dms_id)

        # Preprocess sequences
        print("Preprocessing sequences...")
        sequences = [
            preprocess_sequence(seq)
            for seq in tqdm(
                dms_df["sequence"].tolist(), desc="Preprocessing", unit="seq"
            )
        ]
        print(f"Preprocessed {len(sequences)} sequences")

        # Initialize model
        print("Initializing Evo model...")
        evo_model = Evo(args.model_name)
        model, tokenizer = evo_model.model, evo_model.tokenizer
        model.to(args.device)
        model.eval()

        # Run inference in batches
        print(f"Running inference with batch size {args.batch_size}...")
        logprobs = run_inference(
            model, tokenizer, sequences, args.device, args.batch_size
        )

        # Calculate sequence scores (mean log probability)
        sequence_scores = np.mean(logprobs, axis=1)

        # Add scores to DataFrame
        score_column = f"{args.model_name.replace('-', '_')}_score"
        dms_df[score_column] = sequence_scores

        # Calculate Spearman correlation
        correlation, pvalue = spearmanr(dms_df["DMS_score"], dms_df[score_column])

        # Save results
        output_file = output_dir / f"{dms_id}.csv"
        dms_df.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")

        # Print summary statistics
        print("\nSummary:")
        print(f"Number of sequences: {len(sequences)}")
        print(
            f"Spearman correlation with DMS scores: {correlation:.3f} (p-value: {pvalue:.2e})"
        )
        print(f"Output saved to: {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# python score_evo_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path evo_output --model_name evo-1-8k-base
