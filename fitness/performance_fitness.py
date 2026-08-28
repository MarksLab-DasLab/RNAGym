#!/usr/bin/env python3
"""Evaluate fitness prediction models across RNA assays."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import matthews_corrcoef, roc_auc_score

if __package__:
    from .model_registry import ALL_MODELS
else:
    from model_registry import ALL_MODELS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ASSAY_GROUPS = {
    "ncRNA": ("Ribozyme", "tRNA", "Aptamer"),
    "non-coding": ("Ribozyme", "tRNA", "Aptamer", "mRNA-splicing"),
    "coding": ("mRNA-coding",),
}
DEPTHS = ("single", "multiple")
METRICS = ("Spearman", "AUC", "MCC")
RNA_TYPES = ("mRNA-splicing", "mRNA-coding", "tRNA", "Aptamer", "Ribozyme")


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
        "AUC": roc_auc_score(binary_assay, model_scores),
        "MCC": matthews_corrcoef(binary_assay, binary_model),
    }


def get_performance_dataset(
    data: pd.DataFrame, assay_column: str, score_columns: list[str]
) -> dict[str, dict[str, float]]:
    """Calculate each model's metrics from its finite assay-score pairs."""
    data = data.dropna(subset=["mutant"])
    missing_metrics = dict.fromkeys(METRICS, np.nan)
    results = {}

    for score_column in score_columns:
        if assay_column not in data or score_column not in data:
            results[score_column] = missing_metrics.copy()
            continue

        pairs = data[[assay_column, score_column]].replace([np.inf, -np.inf], np.nan)
        pairs = pairs.dropna()
        if pairs.empty:
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


def _add_depth(data: pd.DataFrame) -> None:
    """Classify mutations without changing the source mutation column."""
    mutation_count = data["mutant"].astype("string").str.count(",") + 1
    data["Depth"] = np.where(mutation_count == 1, "single", "multiple")
    data.loc[mutation_count.isna(), "Depth"] = pd.NA


def _metric_rows(
    dms_id: str,
    rna_type: str,
    results: dict[str, dict[str, float]],
    depth: str | None = None,
) -> list[dict]:
    """Convert one assay's metric dictionaries to rows."""
    rows = []
    for model, metrics in results.items():
        row = {"DMS_ID": dms_id, "RNA_TYPE": rna_type, "Model": model}
        if depth is not None:
            row["Depth"] = depth
        rows.append(row | metrics)
    return rows


def filter_available_models(
    models: list[str], reference: pd.DataFrame, combined_dir: str | Path
) -> list[str]:
    """Drop models that have no score column in any selected assay."""
    available = set()
    combined_dir = Path(combined_dir)

    for dms_id in reference["DMS_ID"]:
        assay_path = combined_dir / f"{dms_id}.csv"
        if not assay_path.exists():
            continue
        columns = pd.read_csv(assay_path, nrows=0).columns
        available.update(model for model in models if f"{model}_score" in columns)
        if len(available) == len(models):
            break

    missing = [model for model in models if model not in available]
    if missing:
        logger.warning("No merged scores found for %s", ", ".join(missing))
    return [model for model in models if model in available]


def filter_msa_assays(
    reference: pd.DataFrame, combined_dir: str | Path
) -> pd.DataFrame:
    """Keep assays with at least ten EVmutation predictions."""
    combined_dir = Path(combined_dir)
    available = []

    for dms_id in reference["DMS_ID"]:
        assay_path = combined_dir / f"{dms_id}.csv"
        if not assay_path.exists():
            continue
        assay = pd.read_csv(assay_path)
        if (
            "EVmutation_score" in assay
            and assay["EVmutation_score"].notna().sum() >= 10
        ):
            available.append(dms_id)

    dropped = len(reference) - len(available)
    if dropped:
        print(f"Dropped {dropped} datasets missing EVmutation scores")
    return reference[reference["DMS_ID"].isin(available)].copy()


def load_assay_metrics(
    reference: pd.DataFrame,
    combined_dir: str | Path,
    score_columns: list[str],
    msa_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate overall and mutation-depth metrics for every selected assay."""
    assay_rows = []
    depth_rows = []
    combined_dir = Path(combined_dir)
    filtered_sizes = []

    for assay_info in reference.itertuples(index=False):
        assay_path = combined_dir / f"{assay_info.DMS_ID}.csv"
        if assay_path.exists():
            assay = pd.read_csv(assay_path)
        else:
            assay = pd.DataFrame(columns=["mutant", "DMS_score", *score_columns])

        if "Depth" not in assay:
            _add_depth(assay)

        filtered = assay
        if msa_only:
            filtered = assay[assay["EVmutation_score"].notna()]
            filtered_sizes.append((len(assay), len(filtered)))

        filtered_results = get_performance_dataset(filtered, "DMS_score", score_columns)
        assay_rows.extend(
            _metric_rows(assay_info.DMS_ID, assay_info.RNA_TYPE, filtered_results)
        )

        overall_results = (
            get_performance_dataset(assay, "DMS_score", score_columns)
            if msa_only
            else filtered_results
        )
        depth_rows.extend(
            _metric_rows(
                assay_info.DMS_ID,
                assay_info.RNA_TYPE,
                overall_results,
                "overall",
            )
        )
        for depth in DEPTHS:
            depth_results = get_performance_dataset(
                assay[assay["Depth"] == depth], "DMS_score", score_columns
            )
            depth_rows.extend(
                _metric_rows(
                    assay_info.DMS_ID,
                    assay_info.RNA_TYPE,
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

    return pd.DataFrame(assay_rows), pd.DataFrame(depth_rows)


def aggregate_by_depth(depth_metrics: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Average assay metrics within each mutation depth."""
    means = {}
    for model in models:
        means[model] = {}
        for depth in DEPTHS:
            assays = depth_metrics[
                (depth_metrics["Model"] == model) & (depth_metrics["Depth"] == depth)
            ]
            means[model][depth] = {
                metric: np.nanmean(assays[metric]) for metric in METRICS
            }
        means[model]["All"] = {
            metric: np.nanmean([means[model][depth][metric] for depth in DEPTHS])
            for metric in METRICS
        }

    rows = []
    for depth in ("All", "multiple", "single"):
        for model in models:
            row = {"Depth": depth, "Model": model}
            row.update(
                {f"{metric}_Mean": means[model][depth][metric] for metric in METRICS}
            )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_rna_type(
    assay_metrics: pd.DataFrame,
    models: list[str],
    rna_types: list[str],
) -> pd.DataFrame:
    """Average assay metrics by RNA type and equally across RNA types."""
    grouped = assay_metrics.groupby(["Model", "RNA_TYPE"], sort=False)[
        list(METRICS)
    ].mean()

    rows = []
    for model in models:
        row = {"Model": model}
        for metric in METRICS:
            type_means = []
            for rna_type in rna_types:
                mean = grouped.loc[(model, rna_type), metric]
                row[f"{metric}_{rna_type}_Mean"] = mean
                type_means.append(mean)
            row[f"{metric}_All_Mean"] = np.mean(type_means)
        rows.append(row)

    columns = ["Model"]
    for metric in METRICS:
        for rna_type in [*rna_types, "All"]:
            columns.append(f"{metric}_{rna_type}_Mean")
    return pd.DataFrame(rows, columns=columns)


def aggregate_by_type_and_depth(
    depth_metrics: pd.DataFrame, models: list[str], rna_types: list[str]
) -> pd.DataFrame:
    """Average metrics by RNA type and mutation depth."""
    grouped = depth_metrics.groupby(["Model", "Depth", "RNA_TYPE"], sort=False)[
        list(METRICS)
    ].mean()
    rows = []

    for model in models:
        for depth in (*DEPTHS, "overall"):
            type_values = []
            for rna_type in rna_types:
                values = grouped.loc[(model, depth, rna_type)]
                row = {"Model": model, "Depth": depth, "RNA_TYPE": rna_type}
                row.update({f"{metric}_Mean": values[metric] for metric in METRICS})
                rows.append(row)
                type_values.append(values)

            all_values = pd.DataFrame(type_values).mean()
            row = {"Model": model, "Depth": depth, "RNA_TYPE": "All"}
            row.update({f"{metric}_Mean": all_values[metric] for metric in METRICS})
            rows.append(row)

    return pd.DataFrame(rows)


def save_results(
    output_dir: str | Path,
    assay_metrics: pd.DataFrame,
    models: list[str],
    by_rna_type: pd.DataFrame,
    by_depth: pd.DataFrame,
    by_type_and_depth: pd.DataFrame,
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
        table.to_csv(output_dir / filename, index=False)

    transposed = assay_metrics[["DMS_ID", "RNA_TYPE"]].drop_duplicates().copy()
    for metric in METRICS:
        metric_table = assay_metrics.pivot(
            index="DMS_ID", columns="Model", values=metric
        )
        for model in models:
            transposed[f"{metric}_{model}"] = transposed["DMS_ID"].map(
                metric_table[model]
            )
    transposed.to_csv(output_dir / "assay_level_results_transposed.csv", index=False)


def select_assays(reference: pd.DataFrame, assay_type: str) -> pd.DataFrame:
    """Select an assay group."""
    if assay_type == "all":
        return reference.copy()
    return reference[reference["RNA_TYPE"].isin(ASSAY_GROUPS[assay_type])].copy()


def main(args) -> None:
    """Run the fitness performance workflow."""
    reference = select_assays(pd.read_csv(args.reference_file), args.type)
    if args.msa_only:
        reference = filter_msa_assays(reference, args.combined_dir)

    models = list(args.models or ALL_MODELS)
    if args.msa_only and "EVmutation" not in models:
        models.append("EVmutation")
    models = filter_available_models(models, reference, args.combined_dir)
    if not models:
        raise ValueError("No model score columns found in the selected assays")

    score_columns = [f"{model}_score" for model in models]
    rna_types = [
        rna_type for rna_type in RNA_TYPES if (reference["RNA_TYPE"] == rna_type).any()
    ]
    assay_metrics, depth_metrics = load_assay_metrics(
        reference, args.combined_dir, score_columns, args.msa_only
    )
    by_rna_type = aggregate_by_rna_type(assay_metrics, score_columns, rna_types)
    by_depth = aggregate_by_depth(depth_metrics, score_columns)
    by_type_and_depth = aggregate_by_type_and_depth(
        depth_metrics, score_columns, rna_types
    )
    save_results(
        args.performance_dir,
        assay_metrics,
        score_columns,
        by_rna_type,
        by_depth,
        by_type_and_depth,
    )

    print("\nMetrics by RNA Type:")
    print(by_rna_type)
    print("\nMetrics by Mutation Depth:")
    print(by_depth)
    print("\nMetrics by RNA Type and Mutation Depth:")
    print(by_type_and_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze fitness predictions across RNA assays"
    )
    parser.add_argument(
        "--reference_file",
        default="reference_sheet_final.csv",
        help="Path to the reference CSV file",
    )
    parser.add_argument(
        "--combined_dir",
        default="merged",
        help="Directory containing merged assay CSV files",
    )
    parser.add_argument(
        "--performance_dir",
        default=None,
        help="Directory for performance tables",
    )
    parser.add_argument(
        "--msa_only",
        action="store_true",
        help="Only evaluate assays with EVmutation scores",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Merged model names to evaluate instead of the defaults",
    )
    parser.add_argument(
        "--type",
        default="all",
        choices=["all", *ASSAY_GROUPS],
        help="Assay group (non-coding includes mRNA-splicing, while ncRNA does not)",
    )
    arguments = parser.parse_args()
    if arguments.performance_dir is None:
        arguments.performance_dir = os.getcwd()
    main(arguments)
