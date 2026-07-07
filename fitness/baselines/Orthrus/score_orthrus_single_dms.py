#!/usr/bin/env python3
"""
Script to run Orthrus inference on DMS assay sequences.
Takes a reference sheet and row ID to process specific assays.

Orthrus (https://github.com/bowang-lab/Orthrus) is a Mamba-based RNA foundation
model. The contrastive checkpoints only produce embeddings, but the
``antichronology/orthrus-mlm-6-track`` checkpoint additionally has a masked
language-modelling head (``predict_tokens`` -> per-position logits over
[A, C, G, T]). We use it to score variants with the masked-marginal
log-likelihood ratio, i.e. the same zero-shot proxy used by the other masked
RNA language-model baselines (RiNALMo, RNA-FM):

    score(variant) = sum_i [ log P(mut_i | context, pos_i masked)
                             - log P(wt_i  | context, pos_i masked) ]

summed over the variant's mutated positions ``i``.

Orthrus MLM is a 6-track model: 4 one-hot nucleotide channels plus a CDS and a
splice channel that are normally derived from a transcript's exon/CDS structure.
DMS constructs are bare sequences without that annotation, so the 2 extra
channels are zero-filled. They are held constant between the wild-type and
mutant bases at each masked position, so the log-likelihood-ratio still isolates
the effect of the sequence change.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from transformers import AutoModel

# One-hot ordering used by Orthrus' seq_to_oh helper.
BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess an RNA/DNA sequence for the Orthrus model:
    - Convert RNA (U) to DNA (T)
    - Convert to uppercase
    - Remove any whitespace
    """
    return sequence.strip().upper().replace("U", "T")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Orthrus (MLM) inference on DMS assay sequences."
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
        default="antichronology/orthrus-mlm-6-track",
        help="Orthrus MLM model to use (default: antichronology/orthrus-mlm-6-track)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of masked positions scored per forward pass (default: 64)",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """
    Load reference sheet and get DMS_ID for specified row.

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


def parse_mutations(mutant_str: str) -> list:
    """
    Parse a mutation string such as ``"A4T,A5G"`` into a list of
    ``(pos0, wt_idx, mut_idx)`` tuples with 0-based positions.

    Raises:
        ValueError: for non-substitution edits (indels) or unknown bases, so
        the caller can score the affected variant as NaN.
    """
    mutations = []
    for token in str(mutant_str).replace(" ", "").split(","):
        if not token:
            continue
        wt_base = token[0].upper().replace("U", "T")
        mut_base = token[-1].upper().replace("U", "T")
        pos = int(token[1:-1]) - 1  # 1-based -> 0-based
        if wt_base not in BASE_TO_IDX or mut_base not in BASE_TO_IDX:
            raise ValueError(f"Unsupported mutation token: {token}")
        mutations.append((pos, BASE_TO_IDX[wt_base], BASE_TO_IDX[mut_base]))
    return mutations


def build_masking_tasks(mutants: list, sequences: list) -> tuple:
    """
    Expand each variant into one masked-scoring task per mutated position.

    Returns:
        tasks: list of ``(row_idx, sequence, pos0, wt_idx, mut_idx)``
        scores: per-variant score array pre-filled with 0.0 for scorable
            variants and NaN for wild-type / unparyseable / out-of-range rows.
    """
    scores = np.full(len(sequences), np.nan, dtype=float)
    tasks = []
    for i, (mutant_str, seq) in enumerate(zip(mutants, sequences)):
        if pd.isna(mutant_str):
            continue  # wild-type row: leave as NaN
        try:
            mutations = parse_mutations(mutant_str)
            if not mutations:
                continue
            row_tasks = []
            for pos, wt_idx, mut_idx in mutations:
                if pos < 0 or pos >= len(seq):
                    raise ValueError(f"Mutation position {pos + 1} outside sequence")
                row_tasks.append((i, seq, pos, wt_idx, mut_idx))
        except (ValueError, IndexError):
            continue  # unsupported edit: leave as NaN
        tasks.extend(row_tasks)
        scores[i] = 0.0  # scorable: accumulate per-position deltas below
    return tasks, scores


def run_inference(
    model, tasks: list, scores: np.ndarray, device: str, batch_size: int
) -> np.ndarray:
    """
    Score masking tasks with Orthrus' masked LM head and accumulate the
    per-position log-likelihood ratios into each variant's score.

    Each task masks a single position (all 6 channels zeroed) and reads the
    predicted log-probabilities of the mutant vs. wild-type base at that
    position. Deltas from the same variant are summed.
    """
    for start in tqdm(
        range(0, len(tasks), batch_size), desc="Scoring", unit="batch"
    ):
        batch = tasks[start : start + batch_size]
        max_len = max(len(seq) for _, seq, _, _, _ in batch)

        # 6-track input: 4 one-hot channels + 2 zero-filled (CDS, splice).
        x = torch.zeros(len(batch), max_len, 6, dtype=torch.float32)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        for b, (_, seq, pos, _, _) in enumerate(batch):
            x[b, : len(seq), :4] = model.seq_to_oh(seq)
            x[b, pos, :] = 0.0  # mask all channels at the scored position
            lengths[b] = len(seq)

        x = x.to(device)
        lengths = lengths.to(device)
        with torch.inference_mode():
            logits = model.predict_tokens(x, lengths, channel_last=True)
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        for b, (row_idx, _, pos, wt_idx, mut_idx) in enumerate(batch):
            delta = (log_probs[b, pos, mut_idx] - log_probs[b, pos, wt_idx]).item()
            scores[row_idx] += delta

    return scores


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

        # Preprocess sequences (RNA -> DNA, uppercase)
        print("Preprocessing sequences...")
        sequences = [preprocess_sequence(seq) for seq in dms_df["sequence"].tolist()]

        # Expand variants into per-position masking tasks
        tasks, scores = build_masking_tasks(dms_df["mutant"].tolist(), sequences)
        print(
            f"Prepared {len(tasks)} masked positions across "
            f"{int(np.sum(~np.isnan(scores)))} scorable variants "
            f"(of {len(sequences)} total)"
        )

        # Initialize model. Orthrus loads via transformers with remote code.
        print(f"Initializing Orthrus model ({args.model_name})...")
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
        model = model.to(args.device)
        model.eval()

        # Run masked-marginal inference
        print(f"Running inference with batch size {args.batch_size}...")
        sequence_scores = run_inference(
            model, tasks, scores, args.device, args.batch_size
        )

        # Add scores to DataFrame
        score_column = "orthrus_score"
        dms_df[score_column] = sequence_scores

        # Calculate Spearman correlation (ignoring unscored variants)
        correlation, pvalue = spearmanr(
            dms_df["DMS_score"], dms_df[score_column], nan_policy="omit"
        )

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

# python score_orthrus_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path orthrus_output --model_name antichronology/orthrus-mlm-6-track
