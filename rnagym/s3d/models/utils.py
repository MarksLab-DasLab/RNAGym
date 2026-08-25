"""Shared utilities for 3D structure prediction adapters."""

import shlex
import shutil
import subprocess
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from rnagym.config import Config3D

Predictor = Callable[[str, Path], Path]


@dataclass(frozen=True)
class Adapter:
    """Hooks used by the shared prediction task."""

    complete: Callable[[Any], bool]
    kinds: tuple[str, ...]
    predict: Callable[[list[Any], str, int], None]
    targets: Callable[[str], list[Any]]
    prepare: Callable[[list[Any], str], None] | None = None
    setup: Callable[[], None] | None = None


def _monomer_complete(model: str, sequence_id: str) -> bool:
    """Check whether one monomer prediction completed."""
    return (_work_dir(model, sequence_id) / "SUCCESS").is_file()


def _predict_monomers(
    model: str,
    predictor: Predictor,
    sequence_ids: list[str],
    _kind: str,
    _shard: int,
) -> None:
    """Run one predictor over independent monomer targets."""
    failures = []
    for sequence_id in sequence_ids:
        work_dir = _work_dir(model, sequence_id)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        try:
            prediction = predictor(sequence_id, work_dir)
            if not prediction.is_file() or prediction.stat().st_size < 100:
                raise RuntimeError(f"Missing or empty prediction: {prediction}")
            (work_dir / "SUCCESS").touch()
        except Exception:
            # Finish independent targets before failing the shard
            traceback.print_exc()
            failures.append(sequence_id)
    if failures:
        raise RuntimeError(f"Failed predictions: {', '.join(failures)}")


def _work_dir(model: str, sequence_id: str) -> Path:
    """Return one monomer prediction directory."""
    return Config3D.PREDICTION_DIR / model / "monomers" / sequence_id


def monomer_adapter(model: str, predictor: Predictor) -> Adapter:
    """Create an adapter for a model that predicts independent monomers."""
    return Adapter(
        complete=partial(_monomer_complete, model),
        kinds=("monomers",),
        predict=partial(_predict_monomers, model, predictor),
        targets=monomer_targets,
    )


def monomer_targets(_kind: str) -> list[str]:
    """Load unique monomer sequences from longest to shortest."""
    return (
        pl.read_parquet(Config3D.TARGET_FILE, columns=["type", "sequence_id", "L"])
        .filter(pl.col("type") == "monomer")
        .group_by("sequence_id")
        .agg(pl.max("L").alias("length"))
        .sort(["length", "sequence_id"], descending=[True, False])
        .get_column("sequence_id")
        .to_list()
    )


def prepare_inputs(
    sequence_id: str,
    work_dir: Path,
    stem: str = "sequence",
    unknown: str = "X",
) -> tuple[Path, Path]:
    """Write the FASTA and A3M inputs shared by monomer models."""
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / f"{stem}.fasta"
    msa, sequence = prepare_msa(sequence_id, work_dir / f"{stem}.a3m", unknown)
    fasta.write_text(f">{sequence_id}\n{sequence}\n")
    return fasta, msa


def prepare_msa(sequence_id: str, output: Path, unknown: str = "A") -> tuple[Path, str]:
    """Write one model MSA and return its path and query sequence."""
    source = Config3D.MSA_DIR / f"{sequence_id}.a3m"
    if not source.is_file():
        raise FileNotFoundError(f"Missing MSA: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    replacements = {"T": "U", "t": "u"}
    if unknown != "X":
        replacements |= {"X": unknown, "x": unknown.lower()}
    text = source.read_text().translate(str.maketrans(replacements))
    output.write_text(text)
    query = []
    for line in text.splitlines()[1:]:
        if line.startswith(">"):
            break
        query.append(line)
    return output.resolve(), "".join(query)


def run(
    command: str,
    cwd: Path | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run one external command."""
    print(f"Running: {command}", flush=True)
    return subprocess.run(
        shlex.split(command),
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def run_ipknot(fasta: Path, output: Path) -> None:
    """Predict the secondary structure used by NuFold and trRosettaRNA."""
    result = run(f"{Config3D.IPKNOT} {fasta}", capture_output=True)
    output.write_text(result.stdout)
