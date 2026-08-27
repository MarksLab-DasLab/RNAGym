"""Shared utilities for 3D structure prediction adapters."""

import math
import shlex
import shutil
import subprocess
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import gemmi
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


def _monomer_complete(model: str, output: str, sequence_id: str) -> bool:
    """Check whether one monomer prediction completed."""
    work_dir = _work_dir(model, sequence_id)
    prediction = work_dir / output.format(sequence_id=sequence_id)
    success = work_dir / "SUCCESS"
    msa = Config3D.MSA_DIR / f"{sequence_id}.a3m"
    return is_fresh(success, (msa,)) and valid_structure(prediction)


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
            if not valid_structure(prediction):
                raise RuntimeError(f"Invalid prediction: {prediction}")
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


def monomer_adapter(model: str, output: str, predictor: Predictor) -> Adapter:
    """Create an adapter for a model that predicts independent monomers."""
    return Adapter(
        complete=partial(_monomer_complete, model, output),
        kinds=("monomers",),
        predict=partial(_predict_monomers, model, predictor),
        targets=monomer_targets,
    )


def is_fresh(path: Path, dependencies: Iterable[Path]) -> bool:
    """Check that one artifact is at least as new as all of its inputs."""
    dependencies = tuple(dependencies)
    return (
        path.is_file()
        and all(dependency.is_file() for dependency in dependencies)
        and path.stat().st_mtime
        >= max(dependency.stat().st_mtime for dependency in dependencies)
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
    table = str.maketrans(replacements)
    text = "".join(
        line if line.startswith(">") else line.translate(table)
        for line in source.read_text().splitlines(keepends=True)
    )
    if not output.is_file() or output.read_text() != text:
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


def valid_structure(path: Path) -> bool:
    """Return whether a prediction contains a usable RNA backbone."""
    if not path.is_file() or path.stat().st_size <= 100:
        return False
    try:
        structure = gemmi.read_structure(str(path))
    except (OSError, RuntimeError, ValueError):
        return False
    atoms = (
        atom
        for model in structure
        for chain in model
        for residue in chain
        for atom in residue
    )
    c3_atoms = 0
    for atom in atoms:
        coordinates = (atom.pos.x, atom.pos.y, atom.pos.z)
        # RhoFold emits +/-999 coordinates when reconstruction fails
        if any(not math.isfinite(value) or abs(value) == 999 for value in coordinates):
            return False
        c3_atoms += atom.name == "C3'"
    return c3_atoms >= 2


def run_ipknot(fasta: Path, output: Path) -> None:
    """Predict the secondary structure used by NuFold and trRosettaRNA."""
    result = run(f"{Config3D.IPKNOT} {fasta}", capture_output=True)
    output.write_text(result.stdout)
