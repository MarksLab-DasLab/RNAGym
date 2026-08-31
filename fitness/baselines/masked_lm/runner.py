"""
Command line entry point shared by the masked language model scorers.

Handles the reference sheet and assay input, the wild-type cross-check, the
strategy selection, and the output file, so that a model's script is only an
adapter.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .engine import accumulate_scores, pad_contexts, window_contexts
from .strategies import STRATEGIES, build_tasks, normalize_strategy, recover_wild_type


def build_parser(adapter) -> argparse.ArgumentParser:
    """Assemble the common arguments plus the adapter's own."""
    parser = argparse.ArgumentParser(
        description=f"Run {adapter.name} masked-marginal inference on DMS assay sequences."
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
        help="Path to reference sheet containing DMS_ID and RAW_CONSTRUCT_SEQ columns",
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
        "--strategies",
        type=str,
        nargs="+",
        default=list(STRATEGIES),
        help="Masked-marginal fill strategies to compute. The fill named is what "
        "the model sees at the variant's OTHER mutated positions while one "
        "position is masked: wt-fill (wild-type bases, what the ESM and "
        "ProteinGym code implement), mask-fill (masks, the ESM paper's "
        "formula), mut-fill (mutant bases), match-fill (the allele being "
        "scored). All four agree on single mutants. Default: all four, which "
        "costs about 19%% more unique context examples than mut-fill alone, "
        "because the contexts are shared",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (default: cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=adapter.default_batch_size,
        help=f"Maximum masked contexts scored per forward pass (default: {adapter.default_batch_size})",
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=adapter.default_max_batch_tokens,
        help="Cap on batch_size x sequence length per forward pass, so that long "
        f"assays automatically use a smaller batch (default: {adapter.default_max_batch_tokens})",
    )
    if adapter.max_tokens is not None:
        parser.add_argument(
            "--max_tokens",
            type=int,
            default=adapter.max_tokens,
            help=f"{adapter.name} position limit including special tokens. Contexts "
            f"longer than this are windowed around the masked span (default: {adapter.max_tokens})",
        )
    parser.add_argument(
        "--quiet_skips",
        action="store_true",
        help="Summarize skipped variants instead of printing one line each",
    )
    adapter.add_arguments(parser)
    return parser


def load_reference_row(ref_sheet_path: str, row_id: int):
    """Return one reference-sheet DMS ID and its optional construct."""
    try:
        ref_df = pd.read_csv(ref_sheet_path, encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found: {ref_sheet_path}")
    if "DMS_ID" not in ref_df.columns:
        raise KeyError("Reference sheet must contain 'DMS_ID' column")
    if not 0 <= row_id < len(ref_df):
        raise ValueError(
            f"Row ID {row_id} is outside the reference sheet's "
            f"0..{len(ref_df) - 1} range"
        )

    dms_id = ref_df.loc[row_id, "DMS_ID"]
    if pd.isna(dms_id):
        raise ValueError(f"DMS_ID is missing for row {row_id}")

    construct = None
    if "RAW_CONSTRUCT_SEQ" in ref_df.columns:
        value = ref_df.loc[row_id, "RAW_CONSTRUCT_SEQ"]
        if not pd.isna(value):
            construct = str(value)
    return str(dms_id), construct


def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """Load and validate one processed assay."""
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")

    df = pd.read_csv(dms_file)
    required_cols = ["mutant", "DMS_score", "sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DMS file: {missing_cols}")

    return df


def resolve_wild_type(adapter, mutants, sequences, construct) -> str:
    """Recover the assay wild type and check it against the reference sheet."""
    recovered = recover_wild_type(mutants, sequences, adapter.bases)
    if construct is None:
        print("Reference sheet has no RAW_CONSTRUCT_SEQ. Using the recovered wild type")
        return recovered
    folded = adapter.canonicalize_sequence(construct)
    if folded != recovered:
        raise ValueError(
            "The wild type recovered from the assay disagrees with the reference "
            f"sheet's RAW_CONSTRUCT_SEQ (lengths {len(recovered)} and {len(folded)}). "
            "The wild-type-background strategies need one agreed background"
        )
    return recovered


def window_budget(adapter, max_tokens: int) -> int:
    """Return the nucleotide budget after declared special tokens."""
    budget = max_tokens - adapter.n_special_tokens
    if budget < 1:
        raise ValueError(
            f"max_tokens={max_tokens} leaves no sequence positions after "
            f"{adapter.n_special_tokens} special tokens"
        )
    return budget


def needs_windowing(contexts, budget: int) -> bool:
    """Whether any context is too long for the model's position limit."""
    return any(len(c) > budget for c in contexts)


def code_revision() -> dict:
    """Return the git revision and a hash of the scoring sources."""
    here = Path(__file__).resolve().parent
    revision, dirty = "unknown", None
    try:
        revision_command = f"git -C {shlex.quote(str(here))} rev-parse HEAD"
        revision = subprocess.run(
            shlex.split(revision_command),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status_command = f"git -C {shlex.quote(str(here))} status --porcelain"
        status = subprocess.run(
            shlex.split(status_command),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    digest = hashlib.sha256()
    sources = sorted(here.glob("*.py"))
    entry = Path(sys.argv[0]).resolve()
    if entry.is_file():
        sources.append(entry)
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "git_head": revision,
        "git_tree_dirty": dirty,
        "scoring_source_sha256": digest.hexdigest(),
        "scoring_source_files": [p.name for p in sources],
    }


def runtime_environment(device) -> dict:
    """Return the Torch, CUDA, and GPU versions used for scoring."""
    import torch

    device = torch.device(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "device": str(device),
        "gpu": device_name,
    }


def write_manifest(path, dms_id, adapter, args, strategies, table, wild_type):
    """Write prediction provenance next to an assay output."""
    manifest = {
        "dms_id": dms_id,
        "environment": runtime_environment(adapter.device),
        "model": adapter.name,
        "score_column_stem": adapter.score_column,
        "strategies": [s.replace("_", "-") for s in strategies],
        "columns": {
            s.replace("_", "-"): f"{adapter.score_column}_{s}" for s in strategies
        },
        "alphabet": adapter.bases,
        "wild_type_length": len(wild_type),
        "context_length": len(table.contexts[0]) if table.contexts else None,
        "context_pad_char": adapter.context_pad_char,
        "contexts": len(table.contexts),
        "terms": table.n_terms(),
        "scorable_variants": int(table.scorable.sum()),
        "code_revision": code_revision(),
        "arguments": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
    }
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(adapter):
    """Run one assay with one model under the requested strategies."""
    parser = build_parser(adapter)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(f"--batch_size must be positive, got {args.batch_size}")
    if args.max_batch_tokens < 1:
        parser.error(
            f"--max_batch_tokens must be positive, got {args.max_batch_tokens}"
        )

    try:
        strategies = tuple(normalize_strategy(s) for s in args.strategies)
    except ValueError as error:
        parser.error(str(error))
    if len(set(strategies)) != len(strategies):
        parser.error(f"Duplicate strategies requested: {args.strategies}")
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dms_id, construct = load_reference_row(args.ref_sheet, args.row_id)
        print(f"Processing DMS ID: {dms_id}")
        dms_df = load_dms_data(args.dms_dir_path, dms_id)
        missing_sequences = dms_df["sequence"].isna()
        if missing_sequences.any():
            rows = dms_df.index[missing_sequences].tolist()[:5]
            raise ValueError(f"DMS file has missing sequences at rows {rows}")

        print(f"Preprocessing sequences into the {adapter.bases} alphabet...")
        sequences = [
            adapter.canonicalize_sequence(s) for s in dms_df["sequence"].tolist()
        ]
        mutants = dms_df["mutant"].tolist()
        wild_type = resolve_wild_type(adapter, mutants, sequences, construct)
        print(f"Wild type: {len(wild_type)} nt, cross-checked against the assay")

        print(
            f"Building contexts for: {', '.join(s.replace('_', '-') for s in strategies)}"
        )
        table = build_tasks(
            mutants,
            sequences,
            wild_type,
            adapter.bases,
            strategies,
            verbose=not args.quiet_skips,
        )
        n_scorable = int(table.scorable.sum())
        print(
            f"Prepared {table.n_terms()} log-probability terms across {n_scorable} "
            f"scorable variants (of {len(sequences)} total), deduplicated to "
            f"{len(table.contexts)} unique contexts"
        )
        if not table.contexts:
            raise ValueError("No scorable variants found")

        max_tokens = getattr(args, "max_tokens", None)
        if max_tokens is not None:
            # The declared count, not the loaded one: this guard runs before the
            # model is loaded so that an unsupported request fails cheaply
            budget = window_budget(adapter, max_tokens)
            if needs_windowing(table.contexts, budget):
                print(
                    f"Windowing contexts to {budget} positions around the masked span"
                )
                window_contexts(table, budget)

        device = args.device
        if device is None:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if adapter.requires_cuda and not device.startswith("cuda"):
            raise ValueError(
                f"{adapter.name} has no working CPU path. Pass a CUDA device"
            )
        args.device = device
        print(f"Initializing {adapter.name} on {device}...")
        adapter.load(args)
        adapter.check_alphabet()

        pad_contexts(table, adapter)
        print("Running inference...")
        scores = accumulate_scores(
            adapter,
            table,
            n_rows=len(dms_df),
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
        )

        for i, strategy in enumerate(strategies):
            dms_df[f"{adapter.score_column}_{strategy}"] = scores[i]
        if len(strategies) == 1:
            dms_df[adapter.score_column] = scores[0]

        print("\nSummary:")
        print(f"Number of sequences: {len(sequences)}")
        for i, strategy in enumerate(strategies):
            correlation, pvalue = spearmanr(
                dms_df["DMS_score"], scores[i], nan_policy="omit"
            )
            print(
                f"  {strategy.replace('_', '-'):10s} Spearman {correlation:+.4f} "
                f"(p {pvalue:.2e})"
            )
        if len(strategies) > 1:
            identical = np.allclose(
                scores[:, table.scorable], scores[0, table.scorable]
            )
            n_multi = 0
            for mutant in np.asarray(mutants)[table.scorable]:
                if str(mutant).count(",") > 0:
                    n_multi += 1
            if n_multi == 0 and not identical:
                raise ValueError(
                    "Every variant is a single mutant, so the strategies must be "
                    "identical, but they are not. This is a bug"
                )
            print(
                f"  strategies identical: {identical} "
                f"({n_multi} multi-mutants among {n_scorable} scorable variants)"
            )

        output_file = output_dir / f"{dms_id}.csv"
        temporary = output_file.with_name(f".{output_file.name}.tmp")
        manifest_file = output_dir / f"{dms_id}.manifest.json"
        manifest_temporary = manifest_file.with_name(f".{manifest_file.name}.tmp")
        dms_df.to_csv(temporary, index=False)
        write_manifest(
            manifest_temporary,
            dms_id,
            adapter,
            args,
            strategies,
            table,
            wild_type,
        )
        temporary.replace(output_file)
        manifest_temporary.replace(manifest_file)
        print(f"Output saved to: {output_file}")

    # Model libraries raise unrelated exception types at this CLI boundary
    except Exception as err:  # noqa: BLE001
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
