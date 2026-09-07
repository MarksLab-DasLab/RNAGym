"""Reproduce fitness inference and the published ncRNA leaderboard."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from scipy.stats import spearmanr

from rnagym.config import ConfigFitness
from rnagym.fitness.data import read_reference
from rnagym.fitness.tasks.merge_scoring_files import standardize_mutation
from rnagym.fitness.tasks.model_registry import (
    ASSAY_GROUPS,
    CHECKPOINT_MODELS,
    FOUR_FILL_SPECS,
    SCORE_COLS,
    STRATEGIES,
    resolve_source,
)

CHECK_ROWS = 128
MIN_CORRELATION = 0.95
MODEL_OVERRIDE_VARIABLES = (
    "AIDO_RNA_MODEL_NAME",
    "AIDO_RNA_PREDICTION_NAME",
    "EVO2_MODEL_NAME",
    "EVO2_LOCAL_PATH",
    "EVO2_PREDICTION_NAME",
    "EVO_MODEL_NAME",
    "EVO_PREDICTION_NAME",
    "NTV3_MODEL_NAME",
    "NTV3_PREDICTION_NAME",
    "ORTHRUS_MODEL_NAME",
)


@dataclass(frozen=True)
class Comparison:
    """Summary of one generated and published prediction comparison."""

    exact: bool
    max_absolute_difference: float
    max_metric_difference: float
    rows: int
    normalized_error: float
    minimum_correlation: float


@dataclass(frozen=True)
class ModelFamily:
    """Prediction launcher, published checkpoints and native scoring test."""

    launcher: str
    checkpoints: tuple[tuple[str, str | None], ...]
    model_variable: str | None = None
    prediction_variable: str | None = None
    native_test: str | None = None


MODEL_FAMILIES = {
    "aido-rna": ModelFamily(
        launcher="baselines/AIDO_RNA/score_aido_rna.sh",
        checkpoints=(
            ("aido_rna_1m", "genbio-ai/AIDO.RNA-1M-MARS"),
            ("aido_rna_25m", "genbio-ai/AIDO.RNA-25M-MARS"),
            ("aido_rna_300m", "genbio-ai/AIDO.RNA-300M-MARS"),
            ("aido_rna_650m", "genbio-ai/AIDO.RNA-650M"),
            ("aido_rna", "genbio-ai/AIDO.RNA-1.6B"),
        ),
        model_variable="AIDO_RNA_MODEL_NAME",
        prediction_variable="AIDO_RNA_PREDICTION_NAME",
    ),
    "evmutation": ModelFamily(
        "baselines/EVmutation/run.sh", (), native_test="test_evmutation_scoring"
    ),
    "evo": ModelFamily(
        launcher="baselines/Evo/score_evo.sh",
        checkpoints=(
            ("evo1", "evo-1-131k-base"),
            ("evo1.5", "evo-1.5-8k-base"),
        ),
        model_variable="EVO_MODEL_NAME",
        prediction_variable="EVO_PREDICTION_NAME",
    ),
    "evo2": ModelFamily(
        launcher="baselines/Evo/score_evo2.sh",
        checkpoints=(
            ("evo2_1b_base", "evo2_1b_base"),
            ("evo2", "evo2_7b"),
            ("evo2_20b", "evo2_20b"),
            ("evo2_40b", "evo2_40b"),
        ),
        model_variable="EVO2_MODEL_NAME",
        prediction_variable="EVO2_PREDICTION_NAME",
    ),
    "genslm": ModelFamily(
        "baselines/GenSLM/run_model.sh",
        (("GenSLM", None),),
        native_test="test_genslm_scoring",
    ),
    "ntv3": ModelFamily(
        launcher="baselines/Nucleotide_Transformer/run.sh",
        checkpoints=(
            ("ntv3_8m", "InstaDeepAI/NTv3_8M_pre"),
            ("ntv3_100m", "InstaDeepAI/NTv3_100M_pre"),
            ("ntv3_650m", "InstaDeepAI/NTv3_650M_pre"),
        ),
        model_variable="NTV3_MODEL_NAME",
        prediction_variable="NTV3_PREDICTION_NAME",
    ),
    "orthrus": ModelFamily("baselines/Orthrus/score_orthrus.sh", (("orthrus", None),)),
    "rna-ernie": ModelFamily("baselines/RNAERNIE/run.sh", (("RNAErnie", None),)),
    "rna-fm": ModelFamily("baselines/RNA_FM/score_rna_fm.sh", (("rna_fm", None),)),
    "rnagenesis": ModelFamily(
        "baselines/RNAGenesis/score_rnagenesis.sh", (("rnagenesis", None),)
    ),
    "rinalmo": ModelFamily("baselines/RiNALMo/score_rinalmo.sh", (("rinalmo", None),)),
}


def check_leaderboard(directory: Path, deadline: float) -> None:
    """Rebuild all 31 ncRNA assays and compare every published category and model."""
    merged, reports = directory / "merged", directory / "reports"
    run_command(
        f"{shlex.quote(sys.executable)} -m rnagym.fitness.tasks.merge_scoring_files --output {shlex.quote(str(merged))}",
        ConfigFitness.REPO_DIR,
        deadline,
    )
    run_command(
        f"{shlex.quote(sys.executable)} -m rnagym.fitness.tasks.leaderboard --input {shlex.quote(str(merged))} --output {shlex.quote(str(reports))}",
        ConfigFitness.REPO_DIR,
        deadline,
    )
    for name in (
        "leaderboard_signed_3ncRNA.csv",
        "leaderboard_evmutation.csv",
        "evmutation_coverage.csv",
    ):
        assert_frame_equal(
            pl.read_csv(reports / name),
            pl.read_csv(ConfigFitness.LEADERBOARD_DIR / name),
            check_exact=False,
            rel_tol=0,
            abs_tol=1e-10,
        )
    print("Both leaderboards and EVmutation coverage matched", flush=True)


def compare_predictions(
    actual_file: Path,
    expected_file: Path,
    score_columns: tuple[str, ...],
) -> Comparison:
    """Require matching rows and score ranks, reporting magnitude and fitness drift."""
    actual = pl.read_csv(actual_file)
    expected = pl.read_csv(expected_file)
    required = ("mutant", "sequence", *score_columns)
    for path, table in ((actual_file, actual), (expected_file, expected)):
        missing = sorted(set(required) - set(table.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

    actual = actual.with_columns(standardize_mutation("mutant"))
    expected = expected.with_columns(standardize_mutation("mutant"))
    for column in ("mutant", "sequence"):
        actual_values = actual.get_column(column).cast(pl.String).to_list()
        expected_values = expected.get_column(column).cast(pl.String).to_list()
        if actual_values != expected_values:
            if len(actual_values) != len(expected_values):
                raise AssertionError(
                    f"{column} row count differs: produced {len(actual_values)}, "
                    f"published {len(expected_values)}"
                )
            row = next(
                index
                for index, values in enumerate(zip(actual_values, expected_values))
                if values[0] != values[1]
            )
            raise AssertionError(
                f"{column} differs at row {row}: produced {actual_values[row]!r}, "
                f"published {expected_values[row]!r}"
            )

    exact = True
    max_metric_difference = 0.0
    normalized_error = 0.0
    minimum_correlation = 1.0
    if actual.is_empty() or not score_columns:
        raise AssertionError("Prediction comparison requires rows and score columns")
    max_absolute_difference = 0.0
    for column in score_columns:
        actual_values = actual.get_column(column).cast(pl.Float64).to_numpy()
        expected_values = expected.get_column(column).cast(pl.Float64).to_numpy()
        if np.isinf(actual_values).any() or np.isinf(expected_values).any():
            raise AssertionError(f"{column} contains an infinite score")

        actual_missing = np.isnan(actual_values)
        expected_missing = np.isnan(expected_values)
        if not np.array_equal(actual_missing, expected_missing):
            row = int(np.flatnonzero(actual_missing != expected_missing)[0])
            mutant = actual.get_column("mutant")[row]
            raise AssertionError(
                f"{column} missing-value status differs for {mutant!r} at row {row}"
            )

        finite = ~actual_missing
        if finite.sum() < 3:
            raise AssertionError(f"{column} has fewer than three finite scores")
        differences = np.abs(actual_values[finite] - expected_values[finite])
        expected_scale = float(np.std(expected_values[finite]))
        if expected_scale == 0:
            raise AssertionError(f"{column}: published scores have no variation")
        if differences.size:
            max_absolute_difference = max(
                max_absolute_difference, float(differences.max())
            )
        column_exact = np.array_equal(actual_values, expected_values, equal_nan=True)
        exact = exact and column_exact
        error = float(np.sqrt(np.mean(differences**2)) / expected_scale)
        if np.ptp(actual_values[finite]) == 0:
            raise AssertionError(
                f"{column}: normalized error {error:.6g}, constant predictions"
            )
        correlation = float(
            spearmanr(actual_values[finite], expected_values[finite]).statistic
        )
        if (
            not np.isfinite(error)
            or not np.isfinite(correlation)
            or correlation < MIN_CORRELATION
        ):
            raise AssertionError(
                f"{column}: normalized error {error:.6g}, rank correlation "
                f"{correlation:.6g} (minimum {MIN_CORRELATION})"
            )
        normalized_error = max(normalized_error, error)
        minimum_correlation = min(minimum_correlation, correlation)
        measurement = "DMS_score" if "DMS_score" in expected else "dms_score"
        if measurement not in expected:
            raise ValueError("Published predictions are missing measured fitness")
        measured = expected[measurement].cast(pl.Float64).to_numpy()[finite]
        if not np.isfinite(measured).all():
            raise AssertionError("Published DMS scores must be finite")
        if np.std(measured) > 0:
            published_metric = spearmanr(measured, expected_values[finite]).statistic
            actual_metric = spearmanr(measured, actual_values[finite]).statistic
            metric_difference = abs(actual_metric - published_metric)
            max_metric_difference = max(max_metric_difference, metric_difference)
    return Comparison(
        exact=exact,
        max_absolute_difference=max_absolute_difference,
        max_metric_difference=max_metric_difference,
        rows=actual.height,
        normalized_error=normalized_error,
        minimum_correlation=minimum_correlation,
    )


def fingerprint() -> str:
    """Hash scoring code, native checks, fixtures and the dependency lock."""
    digest = hashlib.sha256()
    paths = {
        ConfigFitness.REPO_DIR / "pyproject.toml",
        ConfigFitness.REPO_DIR / "rnagym/config.py",
    }
    paths.update((ConfigFitness.REPO_DIR / "rnagym/sh").glob("*.sh"))
    paths.update((ConfigFitness.REPO_DIR / "tests").glob("*fitness*.py"))
    paths.update(
        path
        for path in (ConfigFitness.REPO_DIR / "tests/fixtures/fitness").rglob("*")
        if path.is_file()
    )
    for pattern in (
        "*.py",
        "pixi.*",
        "sh/*.sh",
        "tasks/*.py",
        "baselines/*/*.py",
        "baselines/*/*.patch",
        "baselines/*/*.sh",
        "baselines/*/src/*",
        "baselines/masked_lm/*.py",
    ):
        paths.update(path for path in ConfigFitness.DIR.glob(pattern) if path.is_file())
    for path in sorted(paths):
        digest.update(str(path.relative_to(ConfigFitness.REPO_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    """Require all requested checks to finish before reporting success."""
    args = parse_args(argv)
    if args.timeout <= 0 or not np.isfinite(args.timeout):
        raise ValueError("--timeout must be finite and positive")
    if args.environment and args.leaderboard_only:
        raise ValueError("--leaderboard-only cannot select a model environment")
    started = time.monotonic()
    deadline = started + args.timeout
    report: dict[str, Any] = {
        "status": "running",
        "scope": args.environment
        or ("leaderboard" if args.leaderboard_only else "all"),
        "code_sha256": fingerprint(),
        "environment": {},
        "checks": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Invalidate an earlier success even if this process is killed without cleanup
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    # Forward parent timeouts through the child process groups before exiting
    previous_handler = signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        report["environment"] = runtime_versions()
        if args.environment:
            family = MODEL_FAMILIES[args.environment]
            if family.native_test:
                run_command(
                    f"{shlex.quote(sys.executable)} -m pytest -q tests/fitness_model_checks.py::{family.native_test}",
                    ConfigFitness.REPO_DIR,
                    deadline,
                )
                report["scoring"] = {args.environment: "passed"}
            if family.checkpoints:
                report["checks"].extend(run_one(args.environment, deadline))
        else:
            if not args.leaderboard_only:
                models = [
                    "RNA-FM" if name == "rna_fm" else name
                    for family in MODEL_FAMILIES.values()
                    for name, _ in family.checkpoints
                ]
                if sorted(models) != sorted(CHECKPOINT_MODELS):
                    raise AssertionError(
                        "Reproduction must cover every leaderboard checkpoint exactly once"
                    )
            with tempfile.TemporaryDirectory(prefix="rnagym-reproduce-") as directory:
                root = Path(directory)
                check_leaderboard(root, deadline)
                report["leaderboard"] = "matched"
                if not args.leaderboard_only:
                    for environment in MODEL_FAMILIES:
                        validate_family(environment)
                    for environment in MODEL_FAMILIES:
                        child_report = root / f"{environment}.json"
                        try:
                            run_command(
                                f"pixi run --locked -e {environment} check-published --timeout {deadline - time.monotonic()} --report {shlex.quote(str(child_report))}",
                                ConfigFitness.DIR,
                                deadline,
                            )
                        finally:
                            if child_report.is_file():
                                child = json.loads(child_report.read_text())
                                report["checks"].extend(child["checks"])
                                report.setdefault("scoring", {}).update(
                                    child.get("scoring", {})
                                )
                                report.setdefault("families", {})[environment] = child
                        child = json.loads(child_report.read_text())
                        if child["status"] != "passed":
                            raise AssertionError(f"{environment} did not pass")
        if report["code_sha256"] != fingerprint():
            raise AssertionError("Code or dependencies changed during reproduction")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        report["seconds"] = time.monotonic() - started
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reproduction passed in {report['seconds']:.1f}s: {args.report}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the family, total runtime bound and persistent result report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", nargs="?", choices=tuple(MODEL_FAMILIES))
    parser.add_argument("--leaderboard-only", action="store_true")
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="Total runtime limit in seconds, including child processes",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(tempfile.gettempdir()) / "rnagym-reproduction.json",
    )
    return parser.parse_args(argv)


def prediction_layout(registry_name: str) -> tuple[str, tuple[str, ...]]:
    """Return the published folder and every score column to compare."""
    if registry_name in FOUR_FILL_SPECS:
        folder, column_stem = FOUR_FILL_SPECS[registry_name]
        columns = tuple(f"{column_stem}_{strategy}" for strategy in STRATEGIES)
        return folder, columns
    folder, column = resolve_source(SCORE_COLS, registry_name)
    return folder, (column,)


def prepare_fixture(
    source_data_dir: Path, temporary_data_dir: Path
) -> dict[str, tuple[int, list[int]]]:
    """Keep the complete reference sheet and its original assay row indices."""
    reference_file = source_data_dir / "reference_sheet_final.csv"
    assay_dir = temporary_data_dir / "assays"
    assay_dir.mkdir(parents=True)
    shutil.copy2(reference_file, temporary_data_dir / reference_file.name)
    selections = {}
    for assay_id, row_id in selected_assays(reference_file).items():
        assay = pl.read_csv(source_data_dir / "assays" / f"{assay_id}.csv").drop_nulls(
            "mutant"
        )
        rows = select_rows(assay, CHECK_ROWS)
        assay[rows].write_csv(assay_dir / f"{assay_id}.csv")
        selections[assay_id] = row_id, rows
    return selections


def run_command(
    command: str, cwd: Path, deadline: float, environment: dict[str, str] | None = None
) -> None:
    """Run a command within the suite deadline, terminating its process group on timeout."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Reproduction deadline exceeded")
    with subprocess.Popen(
        shlex.split(command), cwd=cwd, env=environment, start_new_session=True
    ) as process:
        try:
            code = process.wait(timeout=remaining)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def run_one(environment: str, deadline: float) -> Iterator[dict[str, object]]:
    """Exercise the real launcher on distinct original reference-sheet rows."""
    validate_family(environment)
    family = MODEL_FAMILIES[environment]
    with tempfile.TemporaryDirectory(prefix="rnagym-check-published-") as directory:
        temporary_root = Path(directory)
        temporary_data_dir = temporary_root / "fitness"
        selections = prepare_fixture(ConfigFitness.DATA_DIR, temporary_data_dir)
        process_environment = os.environ.copy()
        for variable in MODEL_OVERRIDE_VARIABLES:
            process_environment.pop(variable, None)
        process_environment["RNAGYM_DATA_DIR"] = str(temporary_root)
        launcher = ConfigFitness.DIR / family.launcher
        for registry_name, model_name in family.checkpoints:
            folder, columns = prediction_layout(registry_name)
            if family.model_variable:
                if model_name is None:
                    raise ValueError(f"{registry_name} has no checkpoint name")
                process_environment[family.model_variable] = model_name
            if family.prediction_variable:
                process_environment[family.prediction_variable] = folder
            row_selection = ",".join(str(row) for row, _ in selections.values())
            print(f"Checking {registry_name}: {len(selections)} assays", flush=True)
            started = time.monotonic()
            run_command(
                f"bash {shlex.quote(str(launcher))} {shlex.quote(row_selection)}",
                ConfigFitness.REPO_DIR,
                deadline,
                process_environment,
            )
            checkpoint_seconds = time.monotonic() - started
            for assay_id, (row_id, rows) in selections.items():
                produced = (
                    temporary_data_dir
                    / "model_predictions"
                    / folder
                    / f"{assay_id}.csv"
                )
                if not produced.is_file():
                    raise FileNotFoundError(
                        f"Launcher did not create the selected assay: {produced}"
                    )
                published = pl.read_csv(
                    ConfigFitness.PREDICTION_DIR / folder / f"{assay_id}.csv"
                ).drop_nulls("mutant")
                expected_file = temporary_root / "expected.csv"
                published[rows].write_csv(expected_file)
                try:
                    comparison = compare_predictions(produced, expected_file, columns)
                except (AssertionError, ValueError) as error:
                    saved = Path(
                        tempfile.mkdtemp(prefix=f"rnagym-{registry_name}-{assay_id}-")
                    )
                    shutil.copy2(produced, saved / "actual.csv")
                    shutil.copy2(expected_file, saved / "expected.csv")
                    raise AssertionError(
                        f"{error}. Comparison files saved to {saved}"
                    ) from error
                record = {
                    "environment": environment,
                    "model": registry_name,
                    "assay": assay_id,
                    "row_id": row_id,
                    **comparison.__dict__,
                    "checkpoint_seconds": checkpoint_seconds,
                    "assay_sha256": hashlib.sha256(
                        (temporary_data_dir / "assays" / f"{assay_id}.csv").read_bytes()
                    ).hexdigest(),
                    "published_sha256": hashlib.sha256(
                        expected_file.read_bytes()
                    ).hexdigest(),
                }
                yield record
                produced.unlink()


def runtime_versions() -> dict[str, str]:
    """Record the interpreter and numerical libraries actually used."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "evo-model",
        "evo2",
        "flash-attn",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "numpy",
        "paddlepaddle",
        "polars",
        "scipy",
        "torch",
        "transformer-engine",
        "transformers",
        "vtx",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if "torch" in versions:
        import torch

        versions["cuda"] = torch.version.cuda or "none"
        versions["cudnn"] = str(torch.backends.cudnn.version())
        if torch.cuda.is_available():
            versions["gpu"] = ", ".join(
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            )
    return versions


def selected_assays(reference_file: Path) -> dict[str, int]:
    """Return every ncRNA assay and its original reference-sheet row."""
    reference = read_reference(reference_file).with_row_index("row_id")
    selected = reference.filter(pl.col("RNA_TYPE").is_in(ASSAY_GROUPS["ncRNA"]))
    if selected.is_empty():
        raise ValueError("Reference sheet has no ncRNA assays")
    return dict(selected.select("DMS_ID", "row_id").iter_rows())


def select_rows(table: pl.DataFrame, limit: int | None) -> list[int]:
    """Select reproducible single and multiple mutants in original row order."""
    if limit is None or table.height <= limit:
        return list(range(table.height))
    multiple = table["mutant"].str.contains(",").fill_null(False).to_numpy()
    selected = set()
    for mask in (multiple, ~multiple):
        rows = np.flatnonzero(mask)
        selected.update(
            rows[
                np.linspace(0, len(rows) - 1, min(len(rows), limit // 2), dtype=int)
            ].tolist()
        )
    remaining = sorted(set(range(table.height)) - selected)
    count = min(limit - len(selected), len(remaining))
    if count:
        selected.update(
            np.asarray(remaining)[
                np.linspace(0, len(remaining) - 1, count, dtype=int)
            ].tolist()
        )
    return sorted(selected)


def validate_family(environment: str) -> None:
    """Require every expected checkpoint and assay before loading a model."""
    family = MODEL_FAMILIES[environment]
    assays = selected_assays(ConfigFitness.DATA_DIR / "reference_sheet_final.csv")
    paths = [ConfigFitness.DIR / family.launcher]
    paths.extend(
        ConfigFitness.PREDICTION_DIR / prediction_layout(name)[0] / f"{assay}.csv"
        for name, _ in family.checkpoints
        for assay in assays
    )
    if environment == "genslm":
        paths.append(
            ConfigFitness.CHECKPOINT_DIR
            / "genslm/2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt"
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Reproduction inputs are missing: " + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
