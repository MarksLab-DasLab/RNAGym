#!/usr/bin/env python3
"""
Score RNAGym DMS assays with an Evo 2 model.

Evo 2 is an autoregressive genomic language model (StripedHyena 2). For each
variant sequence we compute the mean per-token log-likelihood under the model
and use it as the fitness score, then report the Spearman correlation against
the experimental ``DMS_score``.

This is the Evo 2 counterpart of ``score_evo_single_dms.py`` (Evo 1 / 1.5). It
uses the official ``evo2`` package (https://github.com/ArcInstitute/evo2) rather
than the ``evo`` package, and supports every checkpoint in ``evo2.utils.MODEL_NAMES``
(``evo2_1b_base``, ``evo2_7b``, ``evo2_20b``, ``evo2_40b``, ...).

Notes on multi-GPU
------------------
Vortex places and (for large models) shards the model across all CUDA devices
that are visible. Select the GPUs with ``CUDA_VISIBLE_DEVICES`` and do NOT move
the model manually with ``.to(device)``. ``evo2_40b`` does not fit on a single
80 GB GPU and needs at least two (e.g. 2xH100-80GB). The 40B/20B/7B/1B
checkpoints all request FP8 via Transformer Engine, i.e. a Hopper GPU.

Offline weights
---------------
The 40B checkpoint ships as two ~41 GB shards that ``evo2`` merges into a single
``evo2_40b.pt`` on first load (a network call). On air-gapped compute nodes,
pre-merge the checkpoint once (see ``download_weights.sh``) and pass the merged
file via ``--local_path`` so no network access is needed at run time.

Batching
--------
All variants of one assay share a single sequence length, so batches never need
padding between sequences and the batch size cannot change a score beyond
floating point roundoff. ``--max_tokens_per_batch`` sizes each batch by a token
budget rather than a sequence count. When FP8 input projections are enabled,
Vortex pads the sequence dimension up to a multiple of 16 inside every
projection, so the budget is applied to that padded length.

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

    # several assays in one process, so the checkpoint is loaded once
    python score_evo2_single_dms.py --row_ids 0-8,11-32 ...
"""

import argparse
import math
import os
import sys
import tempfile
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


def parse_row_ids(spec: str) -> list:
    """Parse a row selection such as ``0-8,11-32,40`` into a sorted list.

    Empty components are rejected rather than skipped: ``0-8,,11`` is far more
    likely to be a typo than an intention, and silently dropping it would score
    a different set of assays than the caller asked for.
    """
    rows = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Empty component in --row_ids: {spec!r}")
        if "-" in part.lstrip("-"):
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if end < start:
                raise ValueError(f"Empty range in --row_ids: {part}")
            rows.update(range(start, end + 1))
        else:
            rows.add(int(part))
    if not rows:
        raise ValueError(f"No rows selected by --row_ids {spec!r}")
    return sorted(rows)


def effective_length(seq_len: int, prepend_bos: bool, fp8: bool) -> int:
    """The sequence length the model actually processes.

    ``prepare_batch`` prepends one token when ``prepend_bos`` is set, and Vortex's
    ``pad_to_multiple`` pads the sequence dimension to a multiple of 16 inside
    every input projection when FP8 is enabled.
    """
    length = seq_len + int(prepend_bos)
    if fp8:
        length = 16 * math.ceil(length / 16)
    return length


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Evo 2 inference on the sequences of one or more DMS assays."
    )
    rows = parser.add_mutually_exclusive_group(required=True)
    rows.add_argument(
        "--row_id",
        type=int,
        help="Row ID in the reference sheet to process",
    )
    rows.add_argument(
        "--row_ids",
        type=str,
        help="Several reference sheet rows, e.g. '0-8,11-32'. They are scored in "
        "one process so the checkpoint is loaded once.",
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
        "max(1, max_tokens_per_batch // effective_length), overriding "
        "--batch_size. The effective length accounts for the BOS token and for "
        "Vortex's multiple-of-16 padding under FP8. Keeps GPU memory roughly "
        "constant across assays of very different lengths (e.g. 8192).",
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
        "--require_fp8",
        action="store_true",
        help="Abort unless the model was actually built with FP8 input "
        "projections. Evo2.load_evo2_model silently falls back to bf16 for 7B "
        "checkpoints when Transformer Engine is unavailable, so without this a "
        "run can be bf16 while everything around it records FP8.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-score even if the output CSV already exists.",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_ids) -> list:
    """Return the DMS_IDs for ``row_ids`` in the reference sheet."""
    try:
        ref_df = pd.read_csv(ref_sheet_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found: {ref_sheet_path}")

    # Tolerate a UTF-8 BOM on the DMS_ID column header.
    ref_df.columns = [c.lstrip("﻿") for c in ref_df.columns]
    if "DMS_ID" not in ref_df.columns:
        raise KeyError("Reference sheet must contain a 'DMS_ID' column")

    dms_ids = []
    for row_id in row_ids:
        if row_id < 0 or row_id >= len(ref_df):
            raise ValueError(
                f"Row ID {row_id} out of range (reference sheet has {len(ref_df)} rows)"
            )
        dms_id = ref_df.loc[row_id, "DMS_ID"]
        if pd.isna(dms_id):
            raise ValueError(f"DMS_ID is missing for row {row_id}")
        dms_ids.append(str(dms_id))
    return dms_ids


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


def write_csv_atomically(df, output_file):
    """Write the scored assay, then rename it into place.

    These CSVs are the published scores, so a partial file must never appear
    under the final name: an interrupted or out-of-quota write would otherwise
    leave a truncated CSV that later looks like a completed assay to the
    resume logic, and --overwrite would destroy a good file to produce it.
    """
    output_file = Path(output_file)
    handle, tmp_path = tempfile.mkstemp(dir=str(output_file.parent),
                                        prefix=f".{output_file.name}.", suffix=".tmp")
    os.close(handle)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_file)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def score_one_assay(evo2_model, args, dms_id, fp8_enabled):
    """Score one assay and write its CSV. Returns the Spearman correlation."""
    output_file = Path(args.output_dir_path) / f"{dms_id}.csv"
    dms_df = load_dms_data(args.dms_dir_path, dms_id)

    # Preprocess sequences (RNA -> DNA), tracking any rows we cannot score.
    print("Preprocessing sequences...")
    raw = dms_df["sequence"]
    valid_mask = raw.notna() & (raw.astype(str).str.strip() != "")
    n_skipped = int((~valid_mask).sum())
    if n_skipped:
        print(f"Skipping {n_skipped} rows with empty/NaN sequence")
    sequences = [preprocess_sequence(s) for s in raw[valid_mask].astype(str)]
    max_len = max((len(s) for s in sequences), default=0)
    print(f"Scoring {len(sequences)} sequences (max length {max_len} nt)")

    # Choose the batch size (token-budget adaptive if requested).
    batch_size = args.batch_size
    if args.max_tokens_per_batch is not None and sequences:
        eff_len = effective_length(max_len, args.prepend_bos, fp8_enabled)
        batch_size = max(1, args.max_tokens_per_batch // eff_len)
        print(f"Token budget {args.max_tokens_per_batch}: seq_len={max_len} "
              f"prepend_bos={args.prepend_bos} fp8={fp8_enabled} "
              f"effective_length={eff_len} -> batch_size={batch_size}")

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

    write_csv_atomically(dms_df, output_file)

    print("\nSummary:")
    print(f"  DMS ID:            {dms_id}")
    print(f"  Sequences scored:  {len(sequences)}")
    print(f"  Score column:      {score_column}")
    print(f"  Spearman vs DMS:   {correlation:.3f} (p={pvalue:.2e})")
    print(f"  Saved to:          {output_file}")
    return correlation


def main():
    args = parse_args()

    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_ids = [args.row_id] if args.row_id is not None else parse_row_ids(args.row_ids)

    dms_ids = load_reference_data(args.ref_sheet, row_ids)
    print(f"Rows {row_ids} -> DMS IDs: {dms_ids}")

    todo = []
    for row_id, dms_id in zip(row_ids, dms_ids):
        output_file = output_dir / f"{dms_id}.csv"
        if output_file.exists() and not args.overwrite:
            print(f"Output already exists (use --overwrite to redo): {output_file}")
            continue
        todo.append((row_id, dms_id))
    if not todo:
        print("Nothing to score.")
        return

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available - Evo 2 requires a GPU.", file=sys.stderr)
    print(f"Visible GPUs: {torch.cuda.device_count()}")

    # Initialize model. Vortex handles device placement / multi-GPU sharding;
    # do NOT call .to(device). The checkpoint is loaded once for every assay.
    print(f"Loading Evo 2 model: {args.model_name} "
          f"(local_path={args.local_path})...")
    evo2_model = Evo2(args.model_name, local_path=args.local_path)
    # Always ask the built model, never the packaged YAML: load_evo2_model can
    # turn FP8 off for 7B when Transformer Engine is missing, and the batch-size
    # arithmetic below has to follow the config the model was actually built with.
    config = evo2_model.model.config
    fp8_enabled = bool(config.get("use_fp8_input_projections", False))
    print(f"use_fp8_input_projections={fp8_enabled}")
    if args.require_fp8 and not fp8_enabled:
        raise SystemExit(
            "--require_fp8 was given but the model resolved to "
            "use_fp8_input_projections=False. For a 7B checkpoint this happens "
            "silently when Transformer Engine is unavailable; for the others it "
            "means Transformer Engine is not providing FP8. Refusing to score, because "
            "the surrounding provenance would claim FP8.")

    failures = []
    for row_id, dms_id in todo:
        print(f"\n=== row {row_id}: {dms_id} ===")
        try:
            score_one_assay(evo2_model, args, dms_id, fp8_enabled)
        except Exception as e:
            print(f"Error scoring {dms_id}: {str(e)}", file=sys.stderr)
            failures.append(dms_id)
            if len(todo) == 1:
                raise

    if failures:
        print(f"\nFAILED assays ({len(failures)}): {failures}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
