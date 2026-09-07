"""Score one assay with all four masked-marginal fill strategies."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import torch

from rnagym.config import ConfigFitness
from rnagym.fitness.baselines.masked_lm.adapter import MaskedLMAdapter
from rnagym.fitness.baselines.masked_lm.engine import (
    accumulate_scores,
    pad_contexts,
    window_contexts,
)
from rnagym.fitness.baselines.masked_lm.strategies import (
    STRATEGIES,
    TaskTable,
    build_tasks,
    recover_wild_type,
)
from rnagym.fitness.data import parse_row_ids, read_assay, read_reference


def build_parser(adapter: MaskedLMAdapter) -> argparse.ArgumentParser:
    """Parse the assay, output directory and checkpoint."""
    parser = argparse.ArgumentParser(description=f"Score an assay with {adapter.name}")
    parser.add_argument(
        "--rows", required=True, help="Reference rows, e.g. 12 or 0-8,12"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Prediction directory"
    )
    adapter.add_arguments(parser)
    return parser


def code_revision() -> dict[str, object]:
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


def load_dms_data(dms_dir_path: str | Path, dms_id: str) -> pl.DataFrame:
    """Load and validate one processed assay."""
    return read_assay(Path(dms_dir_path) / f"{dms_id}.csv")


def load_reference_row(ref_sheet_path: str | Path, row_id: int):
    """Return one reference-sheet DMS ID and its optional construct."""
    reference = read_reference(ref_sheet_path)
    if not 0 <= row_id < reference.height:
        raise ValueError(
            f"Row ID {row_id} is outside the reference sheet's 0..{reference.height - 1} range"
        )
    row = reference.row(row_id, named=True)
    return row["DMS_ID"], row["RAW_CONSTRUCT_SEQ"]


def main(adapter: MaskedLMAdapter):
    """Load one checkpoint and score the selected assays under all fill strategies."""
    parser = build_parser(adapter)
    args = parser.parse_args()

    strategies = STRATEGIES
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        loaded = False
        for row_id in parse_row_ids(args.rows):
            dms_id, construct = load_reference_row(ConfigFitness.REFERENCE_FILE, row_id)
            dms_df = load_dms_data(ConfigFitness.ASSAY_DIR, dms_id)
            missing_sequences = dms_df["sequence"].is_null() | dms_df[
                "sequence"
            ].str.strip_chars().eq("")
            if missing_sequences.any():
                rows = np.flatnonzero(missing_sequences.to_numpy()).tolist()[:5]
                raise ValueError(f"DMS file has missing sequences at rows {rows}")

            sequences = [
                adapter.canonicalize_sequence(s) for s in dms_df["sequence"].to_list()
            ]
            mutants = dms_df["mutant"].to_list()
            wild_type = resolve_wild_type(adapter, mutants, sequences, construct)

            table = build_tasks(
                mutants,
                sequences,
                wild_type,
                adapter.bases,
                strategies,
                verbose=False,
            )
            n_scorable = int(table.scorable.sum())
            if not table.contexts:
                raise ValueError("No scorable variants found")

            max_tokens = adapter.max_tokens
            if max_tokens is not None:
                # The declared count, not the loaded one: this guard runs before the
                # model is loaded so that an unsupported request fails cheaply
                budget = window_budget(adapter, max_tokens)
                if needs_windowing(table.contexts, budget):
                    window_contexts(table, budget)

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if adapter.requires_cuda and not device.startswith("cuda"):
                raise ValueError(f"{adapter.name} requires a CUDA device")
            args.device = device
            if not loaded:
                adapter.load(args)
                adapter.check_alphabet()
                loaded = True

            pad_contexts(table, adapter)
            scores = accumulate_scores(
                adapter,
                table,
                n_rows=len(dms_df),
                batch_size=adapter.default_batch_size,
                max_batch_tokens=adapter.default_max_batch_tokens,
            )

            for i, strategy in enumerate(strategies):
                dms_df = dms_df.with_columns(
                    pl.Series(f"{adapter.score_column}_{strategy}", scores[i])
                )
            multi_mutants = any(
                str(mutant).count(",") > 0
                for mutant in np.asarray(mutants)[table.scorable]
            )
            if not multi_mutants and not np.allclose(
                scores[:, table.scorable], scores[0, table.scorable]
            ):
                raise ValueError("Fill strategies disagree on single mutants")

            output_file = output_dir / f"{dms_id}.csv"
            manifest_file = output_dir / f"{dms_id}.manifest.json"
            with TemporaryDirectory(prefix=f".{dms_id}.", dir=output_dir) as directory:
                temporary = Path(directory) / output_file.name
                manifest_temporary = Path(directory) / manifest_file.name
                dms_df.write_csv(temporary)
                write_manifest(
                    manifest_temporary,
                    temporary,
                    dms_id,
                    adapter,
                    args,
                    strategies,
                    table,
                    wild_type,
                )
                # Keep both outputs from the same writer when assay jobs finish together
                with (output_dir / ".write.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    temporary.replace(output_file)
                    manifest_temporary.replace(manifest_file)
            print(
                f"{dms_id}: {n_scorable}/{len(sequences)} variants scored, saved to {output_file}"
            )

    # Model libraries raise unrelated exception types at this CLI boundary
    except Exception as err:  # noqa: BLE001
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


def needs_windowing(contexts: Sequence[str], budget: int) -> bool:
    """Whether any context is too long for the model's position limit."""
    return any(len(c) > budget for c in contexts)


def resolve_wild_type(
    adapter: MaskedLMAdapter,
    mutants: Sequence[str | None],
    sequences: Sequence[str],
    construct: str | None,
) -> str:
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


def runtime_environment(device: str | torch.device) -> dict[str, object]:
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


def window_budget(adapter: MaskedLMAdapter, max_tokens: int) -> int:
    """Return the nucleotide budget after declared special tokens."""
    budget = max_tokens - adapter.n_special_tokens
    if budget < 1:
        raise ValueError(
            f"max_tokens={max_tokens} leaves no sequence positions after "
            f"{adapter.n_special_tokens} special tokens"
        )
    return budget


def write_manifest(
    path: Path,
    prediction_file: Path,
    dms_id: str,
    adapter: MaskedLMAdapter,
    args: argparse.Namespace,
    strategies: Sequence[str],
    table: TaskTable,
    wild_type: str,
):
    """Write prediction provenance next to an assay output."""
    manifest = {
        "prediction_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
        "dms_id": dms_id,
        "environment": runtime_environment(adapter.device),
        "model": adapter.name,
        "batch_size": adapter.default_batch_size,
        "max_batch_tokens": adapter.default_max_batch_tokens,
        "max_tokens": adapter.max_tokens,
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
