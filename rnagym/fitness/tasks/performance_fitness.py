#!/usr/bin/env python3
"""Evaluate fitness prediction models across RNA assays."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from rnagym.config import ConfigFitness
from rnagym.fitness.data import read_reference
from rnagym.fitness.tasks.model_registry import ALL_MODELS, ASSAY_GROUPS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEPTHS = ("single", "multiple")
METRICS = ("Spearman", "AUC", "MCC")
RNA_TYPES = ("mRNA-splicing", "mRNA-coding", "tRNA", "Aptamer", "Ribozyme")


def _add_depth(data: pl.DataFrame) -> pl.DataFrame:
    """Classify mutations without changing the source mutation column."""
    return data.with_columns(
        pl.when(pl.col("mutant").is_null())
        .then(None)
        .when(pl.col("mutant").str.contains(","))
        .then(pl.lit("multiple"))
        .otherwise(pl.lit("single"))
        .alias("Depth")
    )


def _metric_rows(
    dms_id: str,
    rna_type: str,
    results: dict[str, dict[str, float]],
    depth: str | None = None,
) -> list[dict[str, object]]:
    """Convert one assay's metric dictionaries to rows."""
    rows: list[dict[str, object]] = []
    for model, metrics in results.items():
        row: dict[str, object] = {
            "DMS_ID": dms_id,
            "RNA_TYPE": rna_type,
            "Model": model,
        }
        if depth is not None:
            row["Depth"] = depth
        row.update(metrics)
        rows.append(row)
    return rows


def aggregate_by_depth(depth_metrics: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    """Average assay metrics within each mutation depth."""
    means: dict[str, dict[str, dict[str, float | None]]] = {}
    for model in models:
        means[model] = {}
        for depth in DEPTHS:
            assays = depth_metrics.filter(
                (pl.col("Model") == model) & (pl.col("Depth") == depth)
            )
            means[model][depth] = {
                metric: cast(float | None, assays[metric].mean()) for metric in METRICS
            }
        means[model]["All"] = {
            metric: cast(
                float | None,
                pl.Series(
                    [means[model][depth][metric] for depth in DEPTHS], dtype=pl.Float64
                ).mean(),
            )
            for metric in METRICS
        }

    rows: list[dict[str, object]] = []
    for depth in ("All", "multiple", "single"):
        for model in models:
            row: dict[str, object] = {"Depth": depth, "Model": model}
            row.update(
                {f"{metric}_Mean": means[model][depth][metric] for metric in METRICS}
            )
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def aggregate_by_rna_type(
    assay_metrics: pl.DataFrame,
    models: list[str],
    rna_types: list[str],
) -> pl.DataFrame:
    """Average assay metrics by RNA type and equally across RNA types."""
    grouped = {
        (row["Model"], row["RNA_TYPE"]): row
        for row in assay_metrics.group_by("Model", "RNA_TYPE")
        .agg(pl.col(*METRICS).mean())
        .iter_rows(named=True)
    }

    rows: list[dict[str, object]] = []
    for model in models:
        row: dict[str, object] = {"Model": model}
        for metric in METRICS:
            type_means = []
            for rna_type in rna_types:
                mean = grouped[(model, rna_type)][metric]
                row[f"{metric}_{rna_type}_Mean"] = mean
                type_means.append(mean)
            row[f"{metric}_All_Mean"] = (
                float(np.mean(type_means))
                if all(value is not None for value in type_means)
                else None
            )
        rows.append(row)

    columns = ["Model"]
    for metric in METRICS:
        for rna_type in [*rna_types, "All"]:
            columns.append(f"{metric}_{rna_type}_Mean")
    return pl.DataFrame(rows, infer_schema_length=None).select(columns)


def aggregate_by_type_and_depth(
    depth_metrics: pl.DataFrame, models: list[str], rna_types: list[str]
) -> pl.DataFrame:
    """Average metrics by RNA type and mutation depth."""
    grouped = {
        (row["Model"], row["Depth"], row["RNA_TYPE"]): row
        for row in depth_metrics.group_by("Model", "Depth", "RNA_TYPE")
        .agg(pl.col(*METRICS).mean())
        .iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []

    for model in models:
        for depth in (*DEPTHS, "overall"):
            type_values = []
            for rna_type in rna_types:
                values = grouped[(model, depth, rna_type)]
                row: dict[str, object] = {
                    "Model": model,
                    "Depth": depth,
                    "RNA_TYPE": rna_type,
                }
                row.update({f"{metric}_Mean": values[metric] for metric in METRICS})
                rows.append(row)
                type_values.append(values)

            all_values = (
                pl.DataFrame(type_values, infer_schema_length=None)
                .select(pl.col(*METRICS).mean())
                .row(0, named=True)
            )
            row: dict[str, object] = {"Model": model, "Depth": depth, "RNA_TYPE": "All"}
            row.update({f"{metric}_Mean": all_values[metric] for metric in METRICS})
            rows.append(row)

    return pl.DataFrame(rows, infer_schema_length=None)


def calculate_metrics(
    assay_scores: np.ndarray, model_scores: np.ndarray
) -> dict[str, float]:
    """Calculate directed Spearman, AUC, and MCC.

    AUC binarizes assay scores at their median. MCC binarizes both score vectors
    at their respective medians.
    """
    binary_assay = assay_scores > np.median(assay_scores)
    binary_model = model_scores > np.median(model_scores)
    return {
        "Spearman": stats.spearmanr(assay_scores, model_scores).correlation,
        "AUC": float(roc_auc_score(binary_assay, model_scores)),
        "MCC": matthews_corrcoef(binary_assay, binary_model),
    }


def cli(argv: Sequence[str] | None = None) -> None:
    """Run the fitness performance command."""
    main(parse_args(argv))


def filter_msa_assays(
    reference: pl.DataFrame, combined_dir: str | Path
) -> pl.DataFrame:
    """Keep assays with at least ten EVmutation predictions."""
    combined_dir = Path(combined_dir)
    available = []

    for dms_id in reference["DMS_ID"]:
        assay_path = combined_dir / f"{dms_id}.csv"
        if not assay_path.exists():
            continue
        assay = pl.read_csv(assay_path)
        if (
            "EVmutation_score" in assay
            and assay["EVmutation_score"].is_finite().fill_null(False).sum() >= 10
        ):
            available.append(dms_id)

    dropped = len(reference) - len(available)
    if dropped:
        print(f"Dropped {dropped} datasets missing EVmutation scores")
    return reference.filter(pl.col("DMS_ID").is_in(available))


def get_performance_dataset(
    data: pl.DataFrame, assay_column: str, score_columns: list[str]
) -> dict[str, dict[str, float]]:
    """Calculate each model's metrics from complete finite values."""
    data = data.drop_nulls("mutant")
    missing_metrics = dict.fromkeys(METRICS, np.nan)
    results = {}

    for score_column in score_columns:
        for column in (assay_column, score_column):
            if column not in data:
                raise KeyError(column)
        pairs = data.select(
            pl.col(assay_column, score_column).cast(pl.Float64, strict=False)
        )
        invalid = {
            column: pairs.height - pairs[column].is_finite().fill_null(False).sum()
            for column in pairs.columns
        }
        invalid = {column: count for column, count in invalid.items() if count}
        if invalid:
            raise ValueError(f"Missing or nonfinite metric values: {invalid}")
        if pairs.is_empty():
            results[score_column] = missing_metrics.copy()
            continue

        try:
            results[score_column] = calculate_metrics(
                pairs[assay_column].to_numpy(), pairs[score_column].to_numpy()
            )
        except (TypeError, ValueError) as error:
            logger.warning("Could not calculate %s: %s", score_column, error)
            results[score_column] = missing_metrics.copy()

    return results


def load_assay_metrics(
    reference: pl.DataFrame,
    combined_dir: str | Path,
    score_columns: list[str],
    msa_only: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate overall and mutation-depth metrics for every selected assay."""
    assay_rows = []
    depth_rows = []
    combined_dir = Path(combined_dir)
    filtered_sizes = []

    for assay_info in reference.iter_rows(named=True):
        assay_path = combined_dir / f"{assay_info['DMS_ID']}.csv"
        if not assay_path.is_file():
            raise FileNotFoundError(f"Selected assay is missing: {assay_path}")
        assay = pl.read_csv(assay_path)
        missing_columns = [column for column in score_columns if column not in assay]
        if missing_columns:
            raise KeyError(f"{assay_path} is missing score columns: {missing_columns}")

        if "Depth" not in assay:
            assay = _add_depth(assay)

        filtered = assay
        if msa_only:
            filtered = assay.filter(
                pl.col("EVmutation_score").is_finite().fill_null(False)
            )
            filtered_sizes.append((len(assay), len(filtered)))

        filtered_results = get_performance_dataset(filtered, "DMS_score", score_columns)
        undefined = [
            model
            for model, metrics in filtered_results.items()
            if not np.isfinite(metrics["Spearman"])
        ]
        if undefined:
            raise ValueError(
                f"{assay_path} has undefined Spearman correlation for {undefined}. "
                "Each selected assay requires varying measurements and predictions."
            )
        assay_rows.extend(
            _metric_rows(assay_info["DMS_ID"], assay_info["RNA_TYPE"], filtered_results)
        )

        depth_rows.extend(
            _metric_rows(
                assay_info["DMS_ID"],
                assay_info["RNA_TYPE"],
                filtered_results,
                "overall",
            )
        )
        for depth in DEPTHS:
            depth_results = get_performance_dataset(
                filtered.filter(pl.col("Depth") == depth), "DMS_score", score_columns
            )
            depth_rows.extend(
                _metric_rows(
                    assay_info["DMS_ID"],
                    assay_info["RNA_TYPE"],
                    depth_results,
                    depth,
                )
            )

    if msa_only:
        unfiltered = sum(size[0] for size in filtered_sizes)
        filtered = sum(size[1] for size in filtered_sizes)
        if not unfiltered:
            raise ValueError("No assay rows remain after MSA filtering")
        print(f"Filtered dataset sizes: {[size[1] for size in filtered_sizes]}")
        print(
            f"{filtered / unfiltered * 100:.1f}% of samples remain after "
            "filtering for EVmutation scores"
        )

    return pl.DataFrame(assay_rows).with_columns(
        pl.col(*METRICS).fill_nan(None)
    ), pl.DataFrame(depth_rows).with_columns(pl.col(*METRICS).fill_nan(None))


def main(args: argparse.Namespace) -> None:
    """Run the fitness performance workflow."""
    reference = select_assays(read_reference(ConfigFitness.REFERENCE_FILE), args.type)
    if reference.is_empty():
        raise ValueError(f"No assays found for type {args.type!r}")
    if args.msa_only:
        reference = filter_msa_assays(reference, args.input)
        if reference.is_empty():
            raise ValueError("No assays have at least ten finite EVmutation scores")

    models = list(args.models or ALL_MODELS)
    if args.msa_only and "EVmutation" not in models:
        models.append("EVmutation")

    if len(models) != len(set(models)):
        raise ValueError("Model list contains duplicates")
    score_columns = [f"{model}_score" for model in models]
    rna_types = [
        rna_type for rna_type in RNA_TYPES if (reference["RNA_TYPE"] == rna_type).any()
    ]
    assay_metrics, depth_metrics = load_assay_metrics(
        reference, args.input, score_columns, args.msa_only
    )
    by_rna_type = aggregate_by_rna_type(assay_metrics, score_columns, rna_types)
    by_depth = aggregate_by_depth(depth_metrics, score_columns)
    by_type_and_depth = aggregate_by_type_and_depth(
        depth_metrics, score_columns, rna_types
    )
    save_results(
        args.output,
        assay_metrics,
        score_columns,
        by_rna_type,
        by_depth,
        by_type_and_depth,
    )
    summary = by_rna_type.select(
        "Model",
        *[pl.col(f"Spearman_{name}_Mean").alias(name) for name in rna_types],
        pl.col("Spearman_All_Mean").alias("Mean"),
    ).sort("Mean", descending=True)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(summary)
    print(f"Saved metrics to {args.output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fitness predictions")
    parser.add_argument("--input", type=Path, default=ConfigFitness.COMBINED_DIR)
    parser.add_argument("--output", type=Path, default=ConfigFitness.REPORT_DIR)
    parser.add_argument("--models", nargs="+", help="Models to evaluate")
    parser.add_argument(
        "--msa-only", action="store_true", help="Evaluate MSA-covered variants"
    )
    parser.add_argument("--type", default="ncRNA", choices=["all", *ASSAY_GROUPS])
    return parser.parse_args(argv)


def save_results(
    output_dir: str | Path,
    assay_metrics: pl.DataFrame,
    models: list[str],
    by_rna_type: pl.DataFrame,
    by_depth: pl.DataFrame,
    by_type_and_depth: pl.DataFrame,
) -> None:
    """Write the five benchmark result tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "assay_level_results.csv": assay_metrics,
        "results_by_mutation_depth.csv": by_depth,
        "results_by_rna_type.csv": by_rna_type,
        "results_by_rna_type_and_depth.csv": by_type_and_depth,
    }
    for filename, table in tables.items():
        table.write_csv(output_dir / filename)

    transposed = assay_metrics.select("DMS_ID", "RNA_TYPE").unique(maintain_order=True)
    for metric in METRICS:
        metric_table = assay_metrics.pivot(on="Model", index="DMS_ID", values=metric)
        metric_table = metric_table.select("DMS_ID", *models).rename(
            {model: f"{metric}_{model}" for model in models}
        )
        transposed = transposed.join(
            metric_table, on="DMS_ID", validate="1:1", maintain_order="left"
        )
    transposed.write_csv(output_dir / "assay_level_results_transposed.csv")


def select_assays(reference: pl.DataFrame, assay_type: str) -> pl.DataFrame:
    """Select an assay group."""
    if assay_type == "all":
        return reference.clone()
    return reference.filter(pl.col("RNA_TYPE").is_in(ASSAY_GROUPS[assay_type]))


if __name__ == "__main__":
    cli()
