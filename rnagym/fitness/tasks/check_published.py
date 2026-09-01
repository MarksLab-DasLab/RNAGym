"""Reproduce one released ncRNA assay with each fitness model."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from rnagym.config import ConfigFitness
from rnagym.fitness.tasks.model_registry import (
    FOUR_FILL_SPECS,
    SCORE_COLS,
    STRATEGIES,
    resolve_source,
)

ABSOLUTE_TOLERANCE = 5e-3
ASSAY_ID = "Kobori_2015_ribozyme_p4"
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
RELATIVE_TOLERANCE = 2e-3


@dataclass(frozen=True)
class Comparison:
    """Summary of one generated and published prediction comparison."""

    exact: bool
    max_absolute_difference: float
    rows: int


@dataclass(frozen=True)
class ModelFamily:
    """Prediction launcher and the published checkpoints it supports."""

    launcher: str
    checkpoints: tuple[tuple[str, str | None], ...]
    model_variable: str | None = None
    prediction_variable: str | None = None
    required_variables: tuple[str, ...] = ()


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
        required_variables=("GENSLM_CHECKPOINT_DIR",),
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
    "rna-ernie": ModelFamily(
        "baselines/RNAERNIE/run.sh",
        (("RNAErnie", None),),
        required_variables=("RNAERNIE_CHECKPOINT_DIR",),
    ),
    "rna-fm": ModelFamily(
        "baselines/RNA_FM/score_rna_fm.sh",
        (("rna_fm", None),),
        required_variables=("RNA_FM_CHECKPOINT_PATH",),
    ),
    "rnagenesis": ModelFamily(
        "baselines/RNAGenesis/score_rnagenesis.sh",
        (("rnagenesis", None),),
        required_variables=("RNAGENESIS_MODEL_DIR",),
    ),
    "rinalmo": ModelFamily(
        "baselines/RiNALMo/score_rinalmo.sh",
        (("rinalmo", None),),
        required_variables=("RINALMO_CHECKPOINT_PATH",),
    ),
}


def compare_predictions(
    actual_file: Path, expected_file: Path, score_columns: tuple[str, ...]
) -> Comparison:
    """Require identical rows and numerically equivalent finite scores."""
    actual = pl.read_csv(actual_file)
    expected = pl.read_csv(expected_file)
    required = ("mutant", "sequence", *score_columns)
    for path, table in ((actual_file, actual), (expected_file, expected)):
        missing = sorted(set(required) - set(table.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

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
        differences = np.abs(actual_values[finite] - expected_values[finite])
        if differences.size:
            max_absolute_difference = max(
                max_absolute_difference, float(differences.max())
            )
        column_exact = np.array_equal(actual_values, expected_values, equal_nan=True)
        exact = exact and column_exact
        close = np.isclose(
            actual_values,
            expected_values,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
            equal_nan=True,
        )
        if not close.all():
            row = int(np.flatnonzero(~close)[0])
            mutant = actual.get_column("mutant")[row]
            difference = abs(actual_values[row] - expected_values[row])
            allowed = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(
                expected_values[row]
            )
            raise AssertionError(
                f"{column} differs for {mutant!r} at row {row}: produced "
                f"{actual_values[row]!r}, published {expected_values[row]!r}, "
                f"absolute difference {difference:.6g} exceeds {allowed:.6g}"
            )
    return Comparison(
        exact=exact,
        max_absolute_difference=max_absolute_difference,
        rows=actual.height,
    )


def prediction_layout(registry_name: str) -> tuple[str, tuple[str, ...]]:
    """Return the published folder and every score column to compare."""
    if registry_name in FOUR_FILL_SPECS:
        folder, column_stem = FOUR_FILL_SPECS[registry_name]
        columns = tuple(f"{column_stem}_{strategy}" for strategy in STRATEGIES)
        return folder, columns
    folder, column = resolve_source(SCORE_COLS, registry_name)
    return folder, (column,)


def prepare_fixture(source_data_dir: Path, temporary_data_dir: Path) -> None:
    """Copy the complete check assay and its reference row into a temporary tree."""
    reference_file = source_data_dir / "reference_sheet_final.csv"
    assay_file = source_data_dir / "assays" / f"{ASSAY_ID}.csv"
    for path in (reference_file, assay_file):
        if not path.is_file():
            raise FileNotFoundError(f"Published fitness data is missing: {path}")

    reference = pl.read_csv(reference_file, encoding="utf8-lossy")
    if reference.columns and reference.columns[0].startswith("\ufeff"):
        first = reference.columns[0]
        reference = reference.rename({first: first.lstrip("\ufeff")})
    if "DMS_ID" not in reference:
        raise ValueError(f"{reference_file} is missing column 'DMS_ID'")
    selected = reference.filter(pl.col("DMS_ID") == ASSAY_ID)
    if selected.height != 1:
        raise ValueError(
            f"Expected one {ASSAY_ID} reference row, found {selected.height}"
        )

    assay_dir = temporary_data_dir / "assays"
    assay_dir.mkdir(parents=True)
    (temporary_data_dir / "model_predictions").mkdir()
    selected.write_csv(temporary_data_dir / "reference_sheet_final.csv")
    shutil.copy2(assay_file, assay_dir / assay_file.name)


def validate_family(environment: str) -> None:
    """Fail before inference when inputs or required checkpoint settings are absent."""
    family = MODEL_FAMILIES[environment]
    launcher = ConfigFitness.DIR / family.launcher
    if not launcher.is_file():
        raise FileNotFoundError(f"Prediction launcher is missing: {launcher}")

    missing_variables = [
        variable
        for variable in family.required_variables
        if not os.environ.get(variable)
    ]
    if missing_variables:
        raise RuntimeError(
            f"Set required checkpoint variables: {', '.join(missing_variables)}"
        )

    missing_predictions = []
    for registry_name, _ in family.checkpoints:
        prediction_folder, _ = prediction_layout(registry_name)
        published_file = (
            ConfigFitness.PREDICTION_DIR / prediction_folder / f"{ASSAY_ID}.csv"
        )
        if not published_file.is_file():
            missing_predictions.append(str(published_file))
    if missing_predictions:
        raise FileNotFoundError(
            "Published predictions are missing: " + ", ".join(missing_predictions)
        )


def run_all() -> None:
    """Run each published model environment sequentially."""
    started = time.monotonic()
    missing_variables = sorted(
        {
            variable
            for family in MODEL_FAMILIES.values()
            for variable in family.required_variables
            if not os.environ.get(variable)
        }
    )
    if missing_variables:
        raise RuntimeError(
            f"Set required checkpoint variables: {', '.join(missing_variables)}"
        )
    for environment in MODEL_FAMILIES:
        validate_family(environment)
    for environment in MODEL_FAMILIES:
        print(f"\nChecking {environment}", flush=True)
        command = f"pixi run --locked -e {shlex.quote(environment)} check-published"
        subprocess.run(
            shlex.split(command),
            cwd=ConfigFitness.DIR,
            check=True,
        )
    elapsed = time.monotonic() - started
    model_count = sum(len(family.checkpoints) for family in MODEL_FAMILIES.values())
    print(f"\nAll {model_count} leaderboard models matched in {elapsed:.1f}s")


def run_one(environment: str) -> None:
    """Run one model family against the complete check assay."""
    validate_family(environment)
    family = MODEL_FAMILIES[environment]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rnagym-check-published-") as directory:
        temporary_root = Path(directory)
        temporary_data_dir = temporary_root / "fitness"
        prepare_fixture(ConfigFitness.DATA_DIR, temporary_data_dir)

        process_environment = os.environ.copy()
        for variable in MODEL_OVERRIDE_VARIABLES:
            process_environment.pop(variable, None)
        process_environment["RNAGYM_DATA_DIR"] = str(temporary_root)
        process_environment["SLURM_ARRAY_TASK_ID"] = "0"
        triton_cache = temporary_root / "triton"
        triton_cache.mkdir()
        process_environment["TRITON_CACHE_DIR"] = str(triton_cache)

        launcher = ConfigFitness.DIR / family.launcher
        for registry_name, model_name in family.checkpoints:
            prediction_folder, score_columns = prediction_layout(registry_name)
            published_file = (
                ConfigFitness.PREDICTION_DIR / prediction_folder / f"{ASSAY_ID}.csv"
            )
            if not published_file.is_file():
                raise FileNotFoundError(
                    f"Published predictions are missing: {published_file}"
                )
            if family.model_variable is not None:
                if model_name is None:
                    raise ValueError(
                        f"{registry_name} has no model name for {family.model_variable}"
                    )
                process_environment[family.model_variable] = model_name
            if family.prediction_variable is not None:
                process_environment[family.prediction_variable] = prediction_folder

            print(f"Checking {registry_name}", flush=True)
            command = f"bash {shlex.quote(str(launcher))}"
            subprocess.run(
                shlex.split(command),
                cwd=ConfigFitness.REPO_DIR,
                env=process_environment,
                check=True,
            )

            produced_file = (
                temporary_data_dir
                / "model_predictions"
                / prediction_folder
                / f"{ASSAY_ID}.csv"
            )
            if not produced_file.is_file():
                raise FileNotFoundError(
                    f"Prediction launcher did not create {produced_file}"
                )
            comparison = compare_predictions(
                produced_file,
                published_file,
                score_columns,
            )
            if comparison.exact:
                result = "exact score match"
            else:
                result = (
                    "numerical match, maximum absolute difference "
                    f"{comparison.max_absolute_difference:.6g}"
                )
            print(
                f"{registry_name}: {result} for {comparison.rows} variants "
                f"across {len(score_columns)} score column(s)"
            )

    elapsed = time.monotonic() - started
    print(
        f"{environment}: all {len(family.checkpoints)} checkpoint(s) matched "
        f"{ASSAY_ID} in {elapsed:.1f}s"
    )


def parse_args() -> argparse.Namespace:
    """Parse an optional environment name, defaulting to every model."""
    parser = argparse.ArgumentParser(
        description="Reproduce a released ncRNA assay and verify its scores"
    )
    parser.add_argument("environment", nargs="?", choices=tuple(MODEL_FAMILIES))
    return parser.parse_args()


def main() -> None:
    """Run one environment or the complete published-model check."""
    environment = parse_args().environment
    if environment is None:
        run_all()
    else:
        run_one(environment)


if __name__ == "__main__":
    main()
