#!/usr/bin/env python3
"""Score one or more RNAGym assays with the official Evo 2 predictor.

Multiple ``--rows`` share one model load. Vortex controls model placement
and sharding across the visible GPUs, so this script never moves the model.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

from rnagym.config import ConfigFitness
from rnagym.fitness.data import (
    parse_row_ids,
    read_assay,
    read_reference,
    write_csv_atomically,
)

# FP8 predictions depend on batch shape, so keep the production token budgets
BATCH_TOKENS = {
    "evo2_1b_base": 32768,
    "evo2_7b": 16384,
    "evo2_20b": 16384,
    "evo2_40b": 8192,
}


def effective_length(sequence_length: int, prepend_bos: bool, fp8: bool) -> int:
    """Return the sequence dimension used by the largest input projection."""
    length = sequence_length + int(prepend_bos)
    return ((length + 15) // 16) * 16 if fp8 else length


def load_dms_data(dms_dir: Path, dms_id: str) -> pl.DataFrame:
    """Load and validate an assay table."""
    return read_assay(dms_dir / f"{dms_id}.csv")


def load_reference_data(reference_file: Path, row_ids: list[int]) -> list[str]:
    """Return assay identifiers at validated reference-sheet rows."""
    reference = read_reference(reference_file)
    if any(row < 0 or row >= reference.height for row in row_ids):
        raise ValueError(
            f"Reference rows {row_ids} fall outside 0-{reference.height - 1}"
        )
    return reference[row_ids]["DMS_ID"].to_list()


def main() -> None:
    """Run the Evo 2 scoring command."""
    run(parse_args())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Score assays with Evo 2")
    parser.add_argument(
        "--rows", required=True, help="Reference rows, e.g. 12 or 0-8,12"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", dest="model_name", choices=tuple(BATCH_TOKENS), default="evo2_40b"
    )
    parser.add_argument("--checkpoint", dest="local_path", help="Local checkpoint path")
    return parser.parse_args(argv)


def preprocess_sequence(sequence: str) -> str:
    """Convert an RNA or DNA sequence to uppercase DNA."""
    return sequence.strip().upper().replace("U", "T")


def run(
    args: argparse.Namespace, model_factory: Callable[..., Any] | None = None
) -> None:
    """Load Evo 2 once and score all requested assays."""
    row_ids = parse_row_ids(args.rows)
    dms_ids = load_reference_data(ConfigFitness.REFERENCE_FILE, row_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    pending = list(zip(row_ids, dms_ids))

    if not torch.cuda.is_available():
        print("WARNING: Evo 2 requires CUDA", file=sys.stderr)
    print(f"Loading {args.model_name} on {torch.cuda.device_count()} visible GPUs")
    if model_factory is None:
        from evo2 import Evo2

        model_factory = Evo2
    model = model_factory(args.model_name, local_path=args.local_path)
    fp8 = bool(model.model.config.get("use_fp8_input_projections", False))
    if not fp8:
        raise RuntimeError("The constructed model does not use FP8 input projections")

    failures = []
    for row_id, dms_id in pending:
        try:
            score_assay(model, args, dms_id, fp8)
        except Exception as error:
            if len(pending) == 1:
                raise
            print(f"Row {row_id} ({dms_id}) failed: {error}", file=sys.stderr)
            failures.append(dms_id)
    if failures:
        raise RuntimeError(f"Failed assays: {', '.join(failures)}")


def score_assay(model: Any, args: argparse.Namespace, dms_id: str, fp8: bool) -> float:
    """Score one assay, write its prediction table, and return Spearman rho."""
    data = load_dms_data(ConfigFitness.ASSAY_DIR, dms_id)
    raw_sequences = data["sequence"]
    invalid_sequences = raw_sequences.is_null() | raw_sequences.str.strip_chars().eq("")
    if invalid_sequences.any():
        count = int(invalid_sequences.sum())
        rows = np.flatnonzero(invalid_sequences.to_numpy()).tolist()[:5]
        raise ValueError(
            f"{dms_id} has {count} missing or empty sequences, including rows {rows}"
        )
    sequences = [preprocess_sequence(value) for value in raw_sequences]
    if not sequences:
        raise ValueError(f"{dms_id} has no nonempty sequences")

    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"{dms_id} contains mixed sequence lengths: {sorted(lengths)}")
    sequence_length = lengths.pop()
    length = effective_length(sequence_length, False, fp8)
    batch_size = max(1, BATCH_TOKENS[args.model_name] // length)

    print(
        f"{dms_id}: scoring {len(sequences)} sequences of length {sequence_length} "
        f"in batches of {batch_size}"
    )
    scores = np.asarray(
        model.score_sequences(
            sequences,
            batch_size=batch_size,
            prepend_bos=False,
            reduce_method="mean",
            average_reverse_complement=True,
        ),
        dtype=float,
    )
    if scores.shape != (len(sequences),):
        raise ValueError(
            f"{dms_id} returned score shape {scores.shape}, expected {(len(sequences),)}"
        )
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"{dms_id} returned nonfinite model scores")

    assay_scores = data["DMS_score"].cast(pl.Float64, strict=False).to_numpy()
    if not np.isfinite(assay_scores).all():
        raise ValueError(f"{dms_id} contains missing or nonfinite DMS scores")

    score_column = f"{args.model_name}_score"
    data = data.with_columns(pl.Series(score_column, scores))
    if len(data) < 2:
        correlation = pvalue = float("nan")
    else:
        result = spearmanr(assay_scores, scores)
        correlation, pvalue = result.statistic, result.pvalue

    output_file = args.output / f"{dms_id}.csv"
    write_csv_atomically(data, output_file)
    print(f"{dms_id}: Spearman={correlation:.3f} p={pvalue:.2e} -> {output_file}")
    return correlation


if __name__ == "__main__":
    main()
