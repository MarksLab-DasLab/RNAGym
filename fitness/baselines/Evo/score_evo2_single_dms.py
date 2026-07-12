#!/usr/bin/env python3
"""
Score a single RNAGym DMS assay with an Evo 2 model.

Evo 2 is an autoregressive genomic language model (StripedHyena 2). For each
variant sequence we compute the mean per-token log-likelihood under the model
and use it as the fitness score, then report the Spearman correlation against
the experimental ``DMS_score``.

This is the Evo 2 counterpart of ``score_evo_single_dms.py`` (Evo 1 / 1.5). It
uses the official ``evo2`` package (https://github.com/ArcInstitute/evo2) rather
than the ``evo`` package, and supports the large ``evo2_40b`` checkpoint, which
requires FP8 via Transformer Engine on Hopper GPUs and is automatically sharded
across every visible GPU by the Vortex inference engine.

Notes on multi-GPU
------------------
Vortex places and (for large models) shards the model across all CUDA devices
that are visible. Select the GPUs with ``CUDA_VISIBLE_DEVICES`` and do NOT move
the model manually with ``.to(device)``. ``evo2_40b`` does not fit on a single
80 GB GPU and needs at least two (e.g. 2xH100-80GB or 2xA100-80GB — but note the
40B/20B/1B checkpoints require FP8 + Transformer Engine, i.e. a Hopper GPU).

Offline weights
---------------
The 40B checkpoint ships as two ~41 GB shards that ``evo2`` merges into a single
``evo2_40b.pt`` on first load (a network call). On air-gapped compute nodes,
pre-merge the checkpoint once (see ``download_weights.sh``) and pass the merged
file via ``--local_path`` so no network access is needed at run time.

Usage
-----
    python score_evo2_single_dms.py \
        --row_id 0 \
        --ref_sheet reference_sheet_final.csv \
        --dms_dir_path fitness_processed_assays \
        --output_dir_path evo2_40b_output \
        --model_name evo2_40b \
        --local_path /path/to/evo2_40b.pt \
        --batch_size 1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from evo2 import Evo2


def preprocess_sequence(sequence: str) -> str:
    """Preprocess an RNA/DNA sequence for the Evo 2 (DNA) model.

    - Convert RNA (U) to DNA (T)
    - Uppercase
    - Strip surrounding whitespace
    """
    return sequence.strip().upper().replace("U", "T")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Evo 2 inference on the sequences of a single DMS assay."
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
        help="Path to reference sheet containing a DMS_ID column",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=True,
        help="Directory containing DMS assay CSV files ({DMS_ID}.csv)",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save the scored output CSV",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="evo2_40b",
        help="Evo 2 checkpoint name (default: evo2_40b). The score column is "
        "named '{model_name}_score', e.g. evo2_40b_score.",
    )
    parser.add_argument(
        "--local_path",
        type=str,
        default=None,
        help="Path to a pre-merged Evo 2 .pt checkpoint. When given, the model "
        "is loaded fully offline (no HuggingFace network access). Recommended "
        "for air-gapped compute nodes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sequences scored per forward pass (default: 1). All "
        "variants of an assay share one length, so batching is padding-free; "
        "raise it for short assays, keep it small for very long ones / 40B.",
    )
    parser.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=None,
        help="If set, the batch size is derived per assay as "
        "max(1, max_tokens_per_batch // seq_len), overriding --batch_size. Keeps "
        "GPU memory roughly constant across assays of very different lengths "
        "while maximising throughput (e.g. 8192).",
    )
    parser.add_argument(
        "--reduce_method",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Reduce per-token log-likelihoods by mean (mean PLL, default) or "
        "sum (PLL).",
    )
    parser.add_argument(
        "--prepend_bos",
        action="store_true",
        help="Prepend the BOS/EOD token before scoring (default: off, matching "
        "the evo2 package default).",
    )
    parser.add_argument(
        "--average_reverse_complement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score each sequence as the mean of its forward and "
        "reverse-complement log-likelihood (default: ON). Evo 2 is a "
        "strand-symmetric DNA model and the RNAGym evo2 baselines use "
        "reverse-complement averaging, so this is the default for a fair "
        "comparison. Pass --no-average_reverse_complement for forward strand "
        "only (~2x faster).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-score even if the output CSV already exists.",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """Return the DMS_ID for ``row_id`` in the reference sheet."""
    try:
        ref_df = pd.read_csv(ref_sheet_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found: {ref_sheet_path}")

    # Tolerate a UTF-8 BOM on the DMS_ID column header.
    ref_df.columns = [c.lstrip("﻿") for c in ref_df.columns]
    if "DMS_ID" not in ref_df.columns:
        raise KeyError("Reference sheet must contain a 'DMS_ID' column")
    if row_id < 0 or row_id >= len(ref_df):
        raise ValueError(
            f"Row ID {row_id} out of range (reference sheet has {len(ref_df)} rows)"
        )

    dms_id = ref_df.loc[row_id, "DMS_ID"]
    if pd.isna(dms_id):
        raise ValueError(f"DMS_ID is missing for row {row_id}")
    return str(dms_id)


def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """Load the DMS assay CSV for ``dms_id``."""
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")

    df = pd.read_csv(dms_file)
    required_cols = ["mutant", "DMS_score", "sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DMS file: {missing_cols}")
    return df


def main():
    args = parse_args()

    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dms_id = load_reference_data(args.ref_sheet, args.row_id)
        print(f"Processing DMS ID: {dms_id}")

        output_file = output_dir / f"{dms_id}.csv"
        if output_file.exists() and not args.overwrite:
            print(f"Output already exists (use --overwrite to redo): {output_file}")
            return

        dms_df = load_dms_data(args.dms_dir_path, dms_id)

        # Preprocess sequences (RNA -> DNA), tracking any rows we cannot score.
        print("Preprocessing sequences...")
        raw = dms_df["sequence"]
        valid_mask = raw.notna() & (raw.astype(str).str.strip() != "")
        n_skipped = int((~valid_mask).sum())
        if n_skipped:
            print(f"Skipping {n_skipped} rows with empty/NaN sequence")
        sequences = [preprocess_sequence(s) for s in raw[valid_mask].astype(str)]
        print(f"Scoring {len(sequences)} sequences (max length "
              f"{max((len(s) for s in sequences), default=0)} nt)")

        if not torch.cuda.is_available():
            print("WARNING: CUDA not available — Evo 2 requires a GPU.",
                  file=sys.stderr)
        print(f"Visible GPUs: {torch.cuda.device_count()}")

        # Initialize model. Vortex handles device placement / multi-GPU sharding;
        # do NOT call .to(device).
        print(f"Loading Evo 2 model: {args.model_name} "
              f"(local_path={args.local_path})...")
        evo2_model = Evo2(args.model_name, local_path=args.local_path)

        # Choose the batch size (token-budget adaptive if requested).
        batch_size = args.batch_size
        if args.max_tokens_per_batch is not None and sequences:
            max_len = max(len(s) for s in sequences)
            batch_size = max(1, args.max_tokens_per_batch // max_len)

        print(f"Running inference (batch_size={batch_size}, "
              f"reduce_method={args.reduce_method}, prepend_bos={args.prepend_bos}, "
              f"rc={args.average_reverse_complement})...")
        scores = evo2_model.score_sequences(
            sequences,
            batch_size=batch_size,
            prepend_bos=args.prepend_bos,
            reduce_method=args.reduce_method,
            average_reverse_complement=args.average_reverse_complement,
        )
        scores = np.asarray(scores, dtype=float)

        # Write scores back onto the scored rows (NaN for skipped ones).
        score_column = f"{args.model_name}_score"
        dms_df[score_column] = np.nan
        dms_df.loc[valid_mask, score_column] = scores

        # Spearman on the rows we actually scored.
        scored = dms_df.loc[valid_mask, ["DMS_score", score_column]].dropna()
        if len(scored) >= 2:
            correlation, pvalue = spearmanr(scored["DMS_score"], scored[score_column])
        else:
            correlation, pvalue = float("nan"), float("nan")

        dms_df.to_csv(output_file, index=False)

        print("\nSummary:")
        print(f"  DMS ID:            {dms_id}")
        print(f"  Sequences scored:  {len(sequences)}")
        print(f"  Score column:      {score_column}")
        print(f"  Spearman vs DMS:   {correlation:.3f} (p={pvalue:.2e})")
        print(f"  Saved to:          {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
