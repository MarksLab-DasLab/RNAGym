#!/usr/bin/env python3
"""Score one or more RNAGym assays with the official Evo 2 predictor.

Multiple ``--row_ids`` share one model load. Vortex controls model placement
and sharding across the visible GPUs, so this script never moves the model.
"""

import argparse
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def _nonnegative_int(value: str) -> int:
    """Parse a nonnegative command-line integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    """Parse a positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def effective_length(sequence_length: int, prepend_bos: bool, fp8: bool) -> int:
    """Return the sequence dimension used by the largest input projection."""
    length = sequence_length + int(prepend_bos)
    return ((length + 15) // 16) * 16 if fp8 else length


def load_dms_data(dms_dir: Path, dms_id: str) -> pd.DataFrame:
    """Load and validate an assay table."""
    path = dms_dir / f"{dms_id}.csv"
    data = pd.read_csv(path)
    required = {"mutant", "DMS_score", "sequence"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return data


def load_reference_data(reference_file: Path, row_ids: list[int]) -> list[str]:
    """Return the DMS IDs at the selected reference-sheet rows."""
    reference = pd.read_csv(reference_file, encoding="utf-8-sig")
    if "DMS_ID" not in reference:
        raise ValueError(f"{reference_file} is missing column 'DMS_ID'")

    invalid = [row_id for row_id in row_ids if row_id < 0 or row_id >= len(reference)]
    if invalid:
        raise ValueError(
            f"Reference rows {invalid} fall outside 0-{len(reference) - 1}"
        )
    dms_ids = reference.iloc[row_ids]["DMS_ID"]
    missing = [row_id for row_id, dms_id in zip(row_ids, dms_ids) if pd.isna(dms_id)]
    if missing:
        raise ValueError(f"DMS_ID is missing for reference rows {missing}")
    return dms_ids.astype(str).tolist()


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score one or more RNAGym assays with Evo 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rows = parser.add_mutually_exclusive_group(required=True)
    rows.add_argument(
        "--row_id",
        type=_nonnegative_int,
        help="Reference-sheet row to score",
    )
    rows.add_argument(
        "--row_ids",
        help="Comma-separated rows and ranges to score in one model load, e.g. 0-8,11-32",
    )
    parser.add_argument(
        "--ref_sheet",
        type=Path,
        required=True,
        help="Reference sheet containing DMS_ID",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=Path,
        required=True,
        help="Directory containing assay CSVs",
    )
    parser.add_argument(
        "--output_dir_path",
        type=Path,
        required=True,
        help="Directory for scored CSVs",
    )
    parser.add_argument(
        "--model_name",
        default="evo2_40b",
        help="Evo 2 checkpoint name",
    )
    parser.add_argument(
        "--local_path",
        help="Optional local checkpoint path",
    )
    parser.add_argument(
        "--batch_size",
        type=_positive_int,
        default=1,
        help="Sequences per forward pass",
    )
    parser.add_argument(
        "--max_tokens_per_batch",
        type=_positive_int,
        help="Derive each assay's batch size from this token budget",
    )
    parser.add_argument(
        "--reduce_method",
        choices=("mean", "sum"),
        default="mean",
        help="Per-sequence log-likelihood reduction",
    )
    parser.add_argument(
        "--prepend_bos",
        action="store_true",
        help="Prepend the BOS/EOD token",
    )
    parser.add_argument(
        "--average_reverse_complement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average forward and reverse-complement scores",
    )
    parser.add_argument(
        "--require_fp8",
        action="store_true",
        help="Require FP8 input projections in the constructed model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing scored CSVs",
    )
    return parser.parse_args(argv)


def parse_row_ids(specification: str) -> list[int]:
    """Expand a row selection such as ``0-8,11-32``."""
    rows = set()
    for selection in specification.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:-(\d+))?\s*", selection)
        if match is None:
            raise ValueError(f"Invalid --row_ids selection: {selection!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            raise ValueError(f"Descending --row_ids range: {selection!r}")
        rows.update(range(start, end + 1))
    return sorted(rows)


def preprocess_sequence(sequence: str) -> str:
    """Convert an RNA or DNA sequence to uppercase DNA."""
    return sequence.strip().upper().replace("U", "T")


def score_assay(model, args: argparse.Namespace, dms_id: str, fp8: bool) -> float:
    """Score one assay, write its prediction table, and return Spearman rho."""
    data = load_dms_data(args.dms_dir_path, dms_id)
    raw_sequences = data["sequence"]
    valid = raw_sequences.notna() & raw_sequences.astype(str).str.strip().ne("")
    sequences = [preprocess_sequence(value) for value in raw_sequences[valid]]
    if not sequences:
        raise ValueError(f"{dms_id} has no nonempty sequences")

    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"{dms_id} contains mixed sequence lengths: {sorted(lengths)}")
    sequence_length = lengths.pop()
    batch_size = args.batch_size
    if args.max_tokens_per_batch is not None:
        length = effective_length(sequence_length, args.prepend_bos, fp8)
        batch_size = max(1, args.max_tokens_per_batch // length)

    print(
        f"{dms_id}: scoring {len(sequences)} sequences of length {sequence_length} "
        f"in batches of {batch_size}"
    )
    scores = np.asarray(
        model.score_sequences(
            sequences,
            batch_size=batch_size,
            prepend_bos=args.prepend_bos,
            reduce_method=args.reduce_method,
            average_reverse_complement=args.average_reverse_complement,
        ),
        dtype=float,
    )
    if scores.shape != (len(sequences),):
        raise ValueError(
            f"{dms_id} returned score shape {scores.shape}, expected {(len(sequences),)}"
        )
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"{dms_id} returned nonfinite model scores")

    score_column = f"{args.model_name}_score"
    data[score_column] = np.nan
    data.loc[valid, score_column] = scores
    pairs = data[["DMS_score", score_column]].replace([np.inf, -np.inf], np.nan)
    pairs = pairs.dropna()
    if len(pairs) < 2:
        correlation = pvalue = float("nan")
    else:
        result = spearmanr(pairs["DMS_score"], pairs[score_column])
        correlation, pvalue = result.statistic, result.pvalue

    output_file = args.output_dir_path / f"{dms_id}.csv"
    write_csv_atomically(data, output_file)
    print(f"{dms_id}: Spearman={correlation:.3f} p={pvalue:.2e} -> {output_file}")
    return correlation


def write_csv_atomically(data: pd.DataFrame, output_file: Path) -> None:
    """Replace an output only after its complete CSV has been written."""
    permissions = output_file.stat().st_mode & 0o777 if output_file.exists() else 0o644
    with NamedTemporaryFile(
        dir=output_file.parent,
        prefix=f".{output_file.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_file = Path(handle.name)
    try:
        data.to_csv(temporary_file, index=False)
        temporary_file.chmod(permissions)
        temporary_file.replace(output_file)
    finally:
        temporary_file.unlink(missing_ok=True)


def run(args: argparse.Namespace, model_factory=None) -> None:
    """Load Evo 2 once and score all requested assays."""
    row_ids = [args.row_id] if args.row_id is not None else parse_row_ids(args.row_ids)
    dms_ids = load_reference_data(args.ref_sheet, row_ids)
    args.output_dir_path.mkdir(parents=True, exist_ok=True)
    pending = [
        (row_id, dms_id)
        for row_id, dms_id in zip(row_ids, dms_ids)
        if args.overwrite or not (args.output_dir_path / f"{dms_id}.csv").exists()
    ]
    if not pending:
        print("Nothing to score")
        return

    if not torch.cuda.is_available():
        print("WARNING: Evo 2 requires CUDA", file=sys.stderr)
    print(f"Loading {args.model_name} on {torch.cuda.device_count()} visible GPUs")
    if model_factory is None:
        from evo2 import Evo2

        model_factory = Evo2
    model = model_factory(args.model_name, local_path=args.local_path)
    fp8 = bool(model.model.config.get("use_fp8_input_projections", False))
    if args.require_fp8 and not fp8:
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


def main() -> None:
    """Run the Evo 2 scoring command."""
    run(parse_args())


if __name__ == "__main__":
    main()
