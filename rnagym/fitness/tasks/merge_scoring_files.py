#!/usr/bin/env python3
"""Merge processed DMS assay CSVs with model prediction scores."""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from rnagym.config import ConfigFitness
from rnagym.fitness.data import read_reference
from rnagym.fitness.tasks.model_registry import (
    ALL_MODELS,
    ASSAY_GROUPS,
    SCORE_COLS,
    resolve_source,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def combine_csv_data(
    processed_folder: str | Path,
    model_predictions_folder: str | Path,
    output_folder: str | Path,
    model_list: Sequence[str],
    score_cols_dict: dict[str, str | dict[str, str]],
    assay_types: dict[str, str] | None = None,
    assay_group: str = "all",
):
    """
    Merge predictions without dropping or multiplying assay measurements.

    Every requested model must cover every assay in the selected group.
    The coverage check runs before any output is written, and all merged files
    are staged before they replace prior outputs.
    """
    if len(model_list) != len(set(model_list)):
        raise ValueError(f"Model list contains duplicates: {model_list}")
    processed_files = sorted(Path(processed_folder).glob("*.csv"))
    if not processed_files:
        raise FileNotFoundError(f"No assay CSVs found under {processed_folder}")
    if assay_group != "all" and assay_group not in ASSAY_GROUPS:
        raise ValueError(f"Unknown assay group: {assay_group!r}")
    if assay_types is not None:
        unknown = sorted(
            path.stem for path in processed_files if path.stem not in assay_types
        )
        if unknown:
            raise ValueError(
                f"Processed assays are absent from the reference sheet: {unknown[:5]}"
            )
    if assay_group != "all":
        if assay_types is None:
            raise ValueError("assay_types is required when selecting an assay group")
        selected_types = ASSAY_GROUPS[assay_group]
        processed_files = [
            path for path in processed_files if assay_types[path.stem] in selected_types
        ]
        if not processed_files:
            raise FileNotFoundError(
                f"No {assay_group} assay CSVs found under {processed_folder}"
            )
    processed_names = {path.name for path in processed_files}

    expected = {}
    for model_name in model_list:
        if model_name == "EVmutation":
            expected[model_name] = set()
        else:
            expected[model_name] = processed_names

    missing = {}
    for model_name, assay_names in expected.items():
        folder, _ = resolve_source(score_cols_dict, model_name)
        absent = sorted(
            name
            for name in assay_names
            if not (Path(model_predictions_folder) / folder / name).is_file()
        )
        if absent:
            missing[model_name] = absent
    if missing:
        detail = " | ".join(
            f"{model}: {len(files)} missing, including {files[:3]}"
            for model, files in sorted(missing.items())
        )
        raise FileNotFoundError(
            f"Prediction coverage is incomplete under {model_predictions_folder}: {detail}"
        )

    output_path = Path(output_folder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contributed = {model: set() for model in model_list}

    with tempfile.TemporaryDirectory(
        dir=output_path.parent, prefix=f".{output_path.name}-"
    ) as staging_dir:
        staging_path = Path(staging_dir)
        for processed_file in processed_files:
            csv_file = processed_file.name

            df = pl.read_csv(processed_file)
            mutation_col = get_mutation_column(df)
            df = df.drop_nulls(mutation_col).with_columns(
                standardize_mutation(mutation_col)
            )

            for model_name in model_list:
                folder, score_col = resolve_source(score_cols_dict, model_name)
                model_path = Path(model_predictions_folder) / folder / csv_file
                if not model_path.exists():
                    df = df.with_columns(
                        pl.lit(None, dtype=pl.Float64).alias(f"{model_name}_score")
                    )
                    continue

                df = merge_predictions(
                    df, pl.read_csv(model_path), score_col, model_name
                )
                contributed[model_name].add(csv_file)

            df.write_csv(staging_path / csv_file)

        absent = sorted(model for model, assays in contributed.items() if not assays)
        if absent:
            raise FileNotFoundError(
                f"No predictions found for {absent} under {model_predictions_folder}. "
                "Check the model names and prediction folders."
            )

        output_path.mkdir(parents=True, exist_ok=True)
        staged_names = {path.name for path in staging_path.iterdir()}
        for stale_file in output_path.glob("*.csv"):
            if stale_file.name not in staged_names:
                stale_file.unlink()
        for staged_file in staging_path.iterdir():
            staged_file.replace(output_path / staged_file.name)

    for model_name, assays in sorted(contributed.items()):
        logger.info(f"{model_name}: {len(assays)} assays merged")


def get_mutation_column(df: pl.DataFrame):
    """Find the mutation column name in a DataFrame."""
    for col in ["mutant", "mutation", "Mutation", "mutations"]:
        if col in df.columns:
            return col
    raise ValueError("Couldn't find a mutation column in the dataframe")


def main(argv: Sequence[str] | None = None):
    """Run the prediction merge command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS, help="Models to merge"
    )
    parser.add_argument("--output", type=Path, default=ConfigFitness.COMBINED_DIR)
    parser.add_argument("--type", default="ncRNA", choices=["all", *ASSAY_GROUPS])
    args = parser.parse_args(argv)
    model_list = args.models
    unknown = [m for m in model_list if m not in SCORE_COLS]
    if unknown:
        parser.error(f"No score column configured for: {unknown}")

    reference = read_reference(ConfigFitness.REFERENCE_FILE)
    assay_types = dict(reference.select("DMS_ID", "RNA_TYPE").iter_rows())

    combine_csv_data(
        ConfigFitness.ASSAY_DIR,
        ConfigFitness.PREDICTION_DIR,
        args.output,
        model_list,
        SCORE_COLS,
        assay_types=assay_types,
        assay_group=args.type,
    )


def merge_predictions(
    assay: pl.DataFrame, prediction: pl.DataFrame, score_column: str, model: str
) -> pl.DataFrame:
    """Attach scores while preserving every experimental measurement.

    The published leaderboard uses the first prediction for each mutation and
    retains repeated experimental measurements. Repeated predictions must have
    matching sequences and agree within 0.2% relative or 1e-6 absolute tolerance.
    This allows rounding drift while rejecting conflicting prediction files.
    """
    mutation = get_mutation_column(assay)
    source_mutation = get_mutation_column(prediction)
    if score_column not in prediction:
        raise KeyError(f"{model} has no column {score_column!r}")
    prediction = (
        prediction.drop_nulls(source_mutation)
        .with_columns(standardize_mutation(source_mutation))
        .rename({source_mutation: mutation})
    )
    output_column = f"{model}_score"
    if output_column in assay:
        raise ValueError(f"Assay already contains {output_column}")
    scores = prediction[score_column].cast(pl.Float64, strict=True)
    if model != "EVmutation" and not scores.is_finite().fill_null(False).all():
        raise ValueError(f"{model} has missing or nonfinite predictions")
    if scores.is_infinite().any() or scores.is_nan().any():
        raise ValueError(f"{model} has nonfinite predictions")

    if prediction[mutation].is_duplicated().any():
        for _, repeated in prediction.filter(pl.col(mutation).is_duplicated()).group_by(
            mutation
        ):
            if "sequence" in repeated and repeated["sequence"].n_unique() != 1:
                raise ValueError(f"{model}: repeated prediction sequences conflict")
            values = repeated[score_column].cast(pl.Float64).to_numpy()
            if not np.allclose(values, values[0], rtol=2e-3, atol=1e-6, equal_nan=True):
                raise ValueError(
                    f"{model}: conflicting scores for duplicate mutation {repeated[mutation][0]}"
                )
        prediction = prediction.unique(
            subset=mutation, keep="first", maintain_order=True
        )

    columns = [pl.col(mutation), pl.col(score_column).alias(output_column)]
    if "sequence" in prediction and "sequence" in assay:
        columns.append(pl.col("sequence").alias("_prediction_sequence"))
    prediction = prediction.select(columns)
    matched = assay.join(
        prediction, on=mutation, how="inner", validate="m:1", maintain_order="left"
    )
    if matched.height != assay.height:
        raise ValueError(
            f"Row count mismatch after merging {model}: expected {assay.height}, got {matched.height}"
        )
    if "_prediction_sequence" in matched:
        different = matched.select(
            (
                pl.col("sequence").str.to_uppercase().str.replace_all("T", "U")
                != pl.col("_prediction_sequence")
                .str.to_uppercase()
                .str.replace_all("T", "U")
            )
            .fill_null(True)
            .any()
        ).item()
        if different:
            raise ValueError(f"{model}: prediction sequence disagrees with assay")
        matched = matched.drop("_prediction_sequence")
    return matched


def standardize_mutation(column: str) -> pl.Expr:
    """Normalize a mutation column to uppercase RNA notation."""
    return (
        pl.col(column)
        .str.to_uppercase()
        .str.replace_all("T", "U")
        .str.replace_all(" ", "")
    )


if __name__ == "__main__":
    main()
