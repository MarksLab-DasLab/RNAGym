#!/usr/bin/env python3
"""Evaluate fitness prediction models across RNA assays.

Computes Spearman correlation, AUC, and MCC for each model on each assay,
then aggregates results by RNA type, mutation depth, and their combination.
"""

import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

METRICS = ["Spearman", "AUC", "MCC"]


# --- Core metric computation ---


def calculate_metrics(
    assay_scores: np.ndarray, model_scores: np.ndarray
) -> Dict[str, float]:
    """Calculate Spearman correlation, AUC, and MCC."""
    spearman_corr = stats.spearmanr(assay_scores, model_scores).correlation
    binary_true = (assay_scores > np.median(assay_scores)).astype(int)
    binary_pred = (model_scores > np.median(model_scores)).astype(int)
    auc = roc_auc_score(y_true=binary_true, y_score=model_scores)
    mcc = matthews_corrcoef(y_true=binary_true, y_pred=binary_pred)
    return {"Spearman": abs(spearman_corr), "AUC": max(auc, 1 - auc), "MCC": abs(mcc)}


def get_performance_dataset(
    df: pd.DataFrame, assay_col: str, score_columns: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each model on a single assay DataFrame."""
    df = df.dropna(subset=["mutant"])
    nan_result = {m: np.nan for m in METRICS}
    results = {}
    for model in score_columns:
        try:
            subset = df.dropna(subset=[assay_col, model])
        except KeyError:
            subset = pd.DataFrame()
        if subset.empty:
            results[model] = nan_result.copy()
        else:
            try:
                results[model] = calculate_metrics(
                    subset[assay_col].values, subset[model].values
                )
            except Exception as e:
                logger.warning(f"Error calculating metrics for {model}: {e}")
                results[model] = nan_result.copy()
    return results


# --- Helpers ---


def _add_depth_column(df: pd.DataFrame) -> None:
    """Classify mutations as single or multiple in place."""
    df["mutant"] = df["mutant"].astype(str)
    df["mutation_count"] = df["mutant"].apply(lambda x: len(x.split(",")))
    df["Depth"] = np.where(df["mutation_count"] == 1, "single", "multiple")


def _store_metrics(wt_seqs, dataset, results, score_columns, suffix=""):
    """Write per-assay metric results into wt_seqs."""
    for model in score_columns:
        for metric in METRICS:
            wt_seqs.loc[wt_seqs["DMS_ID"] == dataset, f"{metric}_{model}{suffix}"] = (
                results[model][metric]
            )


# --- Data loading ---


def filter_available_models(
    model_list: List[str], wt_seqs: pd.DataFrame, combined_dir: str
) -> List[str]:
    """Drop models that have no score column in any of the merged assay files.

    Predictions are added to RNAGym one model at a time, and a model can also
    cover only part of the benchmark. Scoring a model that was never merged in
    would add an all-NaN row to every output table, so it is dropped instead.
    """
    present = set()
    for dataset in wt_seqs["DMS_ID"]:
        df_path = f"{combined_dir}/{dataset}.csv"
        if not os.path.exists(df_path):
            continue
        columns = set(pd.read_csv(df_path, nrows=0).columns)
        present.update(m for m in model_list if f"{m}_score" in columns)
        if len(present) == len(model_list):
            break

    dropped = [m for m in model_list if m not in present]
    if dropped:
        logger.warning(f"No merged scores found for: {', '.join(dropped)}. Skipping.")
    return [m for m in model_list if m in present]


def load_assay_metrics(
    wt_seqs: pd.DataFrame,
    combined_dir: str,
    score_columns: List[str],
    msa_only: bool = False,
) -> pd.DataFrame:
    """Load merged CSVs once and compute all per-assay metrics.

    Populates wt_seqs with:
      {Metric}_{model}          – overall (filtered when msa_only)
      {Metric}_{model}_{depth}  – per single/multiple depth (unfiltered)
      {Metric}_{model}_overall  – unfiltered overall (for combined table)
    """
    n_unfiltered, n_filtered = [], []

    for _, row in wt_seqs.iterrows():
        dataset = row["DMS_ID"]
        df_path = f"{combined_dir}/{dataset}.csv"
        if not os.path.exists(df_path):
            continue

        df = pd.read_csv(df_path)

        # Add depth column if missing, persist to CSV
        if "Depth" not in df.columns:
            _add_depth_column(df)
            df.to_csv(df_path, index=False)

        # Overall metrics (filtered when msa_only, used by type table + assay outputs)
        if msa_only:
            n_unfiltered.append(len(df))
            df_filtered = df[~df["EVmutation_score"].isna()]
            n_filtered.append(len(df_filtered))
        else:
            df_filtered = df

        results_filtered = get_performance_dataset(
            df_filtered, "DMS_score", score_columns
        )
        _store_metrics(wt_seqs, dataset, results_filtered, score_columns)

        # Unfiltered overall metrics (for combined table's "overall" depth)
        if msa_only:
            results_unfiltered = get_performance_dataset(df, "DMS_score", score_columns)
        else:
            results_unfiltered = results_filtered
        _store_metrics(
            wt_seqs, dataset, results_unfiltered, score_columns, suffix="_overall"
        )

        # Per-depth metrics (always unfiltered)
        for depth in ["single", "multiple"]:
            subset = df[df["Depth"] == depth]
            if not subset.empty:
                depth_results = get_performance_dataset(
                    subset, "DMS_score", score_columns
                )
                _store_metrics(
                    wt_seqs, dataset, depth_results, score_columns, suffix=f"_{depth}"
                )

    if msa_only:
        pct = sum(n_filtered) / sum(n_unfiltered) * 100.0
        print(f"Filtered dataset sizes: {n_filtered}")
        print(f"{pct:.1f}% of samples remain after filtering for EVmutation scores")

    return wt_seqs


# --- Bootstrap SE ---


def bootstrap_se(
    data: pd.DataFrame,
    group_col: str,
    metric_col: str,
    types: List[str],
    score_columns: List[str],
    number_assay_reshuffle: int = 10000,
) -> Dict[str, float]:
    """Bootstrap SE of differences from the best model, per group and overall."""
    best_models = data.groupby(group_col)[score_columns].mean().idxmax(axis=1)
    best_model_all = data[score_columns].mean().idxmax()
    bootstrap_means = {t: [] for t in types + ["All"]}

    for _ in range(number_assay_reshuffle):
        resampled = data.sample(frac=1.0, replace=True)
        resampled_groups = resampled.groupby(group_col)

        for type_ in types:
            if type_ in resampled_groups.groups:
                group_data = resampled_groups.get_group(type_)
                if not group_data.empty:
                    diffs = group_data[metric_col] - group_data[best_models[type_]]
                    bootstrap_means[type_].append(diffs.mean())

        all_diffs = resampled[metric_col] - resampled[best_model_all]
        bootstrap_means["All"].append(all_diffs.mean())

    se = {}
    for type_ in types + ["All"]:
        values = [v for v in bootstrap_means[type_] if not np.isnan(v)]
        se[type_] = np.std(values, ddof=1) if len(values) > 1 else np.nan
    return se


def bootstrap_se_of_means(
    df: pd.DataFrame,
    type_col: str,
    metric_col: str,
    types: List[str],
    metric_columns: List[str],
    n_iterations: int,
) -> float:
    """Bootstrap SE of the mean-of-type-means, with assay reshuffling."""
    other_scores = df[metric_columns].mean()
    if all(other_scores[metric_col] >= other_scores):
        return 0.0

    bootstrap_means = []
    for _ in range(n_iterations):
        reshuffled_df = df.copy()
        metric_data = reshuffled_df[metric_columns].values
        np.random.shuffle(metric_data)
        reshuffled_df[metric_columns] = metric_data

        type_means = []
        for rna_type in types:
            type_data = reshuffled_df[reshuffled_df[type_col] == rna_type][metric_col]
            if len(type_data) > 0:
                bootstrap_sample = type_data.sample(n=len(type_data), replace=True)
                type_means.append(bootstrap_sample.mean())
        if type_means:
            bootstrap_means.append(np.mean(type_means))

    return np.std(bootstrap_means)


# --- Aggregation ---


def aggregate_by_rna_type(
    wt_seqs: pd.DataFrame,
    types: List[str],
    score_columns: List[str],
    calculate_se: bool,
    number_assay_reshuffle: int = 1000,
) -> pd.DataFrame:
    """Average metrics by RNA type. 'All' = mean of type means."""
    rows = []
    for model in score_columns:
        row = {"Model": model}
        for metric in METRICS:
            col = f"{metric}_{model}"
            type_means = {}
            for rna_type in types:
                mean_val = wt_seqs[wt_seqs["RNA_TYPE"] == rna_type][col].mean()
                type_means[rna_type] = mean_val
                row[f"{metric}_{rna_type}_Mean"] = mean_val
            row[f"{metric}_All_Mean"] = sum(type_means.values()) / len(type_means)

            if calculate_se:
                metric_columns = [f"{metric}_{c}" for c in score_columns]
                se = bootstrap_se(
                    wt_seqs,
                    "RNA_TYPE",
                    col,
                    types,
                    metric_columns,
                    number_assay_reshuffle,
                )
                for rna_type in types + ["All"]:
                    row[f"{metric}_{rna_type}_SE"] = se[rna_type]
        rows.append(row)

    result = pd.DataFrame(rows)
    col_order = ["Model"]
    for metric in METRICS:
        for rna_type in types + ["All"]:
            col_order.append(f"{metric}_{rna_type}_Mean")
            if calculate_se:
                col_order.append(f"{metric}_{rna_type}_SE")
    return result[col_order]


def aggregate_by_depth(
    wt_seqs: pd.DataFrame,
    score_columns: List[str],
    calculate_se: bool = False,
    number_assay_reshuffle: int = 10000,
) -> pd.DataFrame:
    """Average metrics by mutation depth. 'All' = mean of depth means."""
    means = {}
    se_vals = {}
    for model in score_columns:
        means[model] = {}
        se_vals[model] = {}
        for metric in METRICS:
            depth_means = {}
            depth_ses = {}
            for depth in ["single", "multiple"]:
                depth_means[depth] = np.nanmean(wt_seqs[f"{metric}_{model}_{depth}"])
                if calculate_se:
                    metric_cols = [f"{metric}_{c}_{depth}" for c in score_columns]
                    se = bootstrap_se(
                        wt_seqs,
                        "DMS_ID",
                        f"{metric}_{model}_{depth}",
                        [depth],
                        metric_cols,
                        number_assay_reshuffle,
                    )
                    depth_ses[depth] = se[depth]
            depth_means["All"] = np.nanmean(
                [depth_means["single"], depth_means["multiple"]]
            )
            if calculate_se:
                depth_ses["All"] = np.sqrt(
                    (depth_ses["single"] ** 2 + depth_ses["multiple"] ** 2) / 2
                )
            means[model][metric] = depth_means
            se_vals[model][metric] = depth_ses

    rows = []
    for depth in ["single", "multiple", "All"]:
        for model in score_columns:
            row = {"Depth": depth, "Model": model}
            for metric in METRICS:
                row[f"{metric}_Mean"] = means[model][metric][depth]
                if calculate_se:
                    row[f"{metric}_SE"] = se_vals[model][metric][depth]
            rows.append(row)
    return pd.DataFrame(rows).sort_values("Depth")


def aggregate_by_type_and_depth(
    wt_seqs: pd.DataFrame,
    types: List[str],
    score_columns: List[str],
    calculate_se: bool = False,
    number_assay_reshuffle: int = 10000,
) -> pd.DataFrame:
    """Average metrics by RNA type x mutation depth (including 'overall')."""
    results = []
    for model in score_columns:
        for depth in ["single", "multiple", "overall"]:
            suffix = f"_{depth}"

            # Per RNA type
            for rna_type in types:
                type_data = wt_seqs[wt_seqs["RNA_TYPE"] == rna_type]
                row = {"Model": model, "Depth": depth, "RNA_TYPE": rna_type}
                for metric in METRICS:
                    col = f"{metric}_{model}{suffix}"
                    values = type_data[col].dropna()
                    row[f"{metric}_Mean"] = (
                        np.mean(values) if len(values) > 0 else np.nan
                    )
                    if calculate_se:
                        if len(values) > 0:
                            metric_cols = [
                                f"{metric}_{c}{suffix}" for c in score_columns
                            ]
                            se = bootstrap_se(
                                type_data,
                                "DMS_ID",
                                col,
                                [rna_type],
                                metric_cols,
                                number_assay_reshuffle,
                            )
                            row[f"{metric}_SE"] = se[rna_type]
                        else:
                            row[f"{metric}_SE"] = np.nan
                results.append(row)

            # "All" = mean of type means
            row = {"Model": model, "Depth": depth, "RNA_TYPE": "All"}
            for metric in METRICS:
                type_means = []
                for rna_type in types:
                    col = f"{metric}_{model}{suffix}"
                    values = wt_seqs[wt_seqs["RNA_TYPE"] == rna_type][col].dropna()
                    if len(values) > 0:
                        type_means.append(np.mean(values))
                if type_means:
                    row[f"{metric}_Mean"] = np.mean(type_means)
                    if calculate_se:
                        ses = []
                        for rna_type in types:
                            type_data = wt_seqs[wt_seqs["RNA_TYPE"] == rna_type]
                            if type_data.empty:
                                continue
                            col = f"{metric}_{model}{suffix}"
                            metric_cols = [
                                f"{metric}_{c}{suffix}" for c in score_columns
                            ]
                            se = bootstrap_se(
                                type_data,
                                "DMS_ID",
                                col,
                                [rna_type],
                                metric_cols,
                                number_assay_reshuffle,
                            )
                            if not np.isnan(se[rna_type]):
                                ses.append(se[rna_type] ** 2)
                        row[f"{metric}_SE"] = np.sqrt(np.mean(ses)) if ses else np.nan
                else:
                    row[f"{metric}_Mean"] = np.nan
                    if calculate_se:
                        row[f"{metric}_SE"] = np.nan
            results.append(row)

    return pd.DataFrame(results)


# --- Output ---


def save_assay_level_results(
    wt_seqs: pd.DataFrame, score_columns: List[str], output_file: str
):
    """Save assay-level results with one row per (assay, model)."""
    data = []
    for _, row in wt_seqs.iterrows():
        for model in score_columns:
            data.append(
                {
                    "DMS_ID": row["DMS_ID"],
                    "RNA_TYPE": row["RNA_TYPE"],
                    "Model": model,
                    "Spearman": row[f"Spearman_{model}"],
                    "AUC": row[f"AUC_{model}"],
                    "MCC": row[f"MCC_{model}"],
                }
            )
    pd.DataFrame(data).to_csv(output_file, index=False)


def save_assay_level_results_transposed(
    wt_seqs: pd.DataFrame, score_columns: List[str], output_file: str
):
    """Save assay-level results grouped by metric type."""
    cols = ["DMS_ID", "RNA_TYPE"]
    for metric in METRICS:
        for model in score_columns:
            cols.append(f"{metric}_{model}")
    cols = [c for c in cols if c in wt_seqs.columns]
    wt_seqs.to_csv(output_file, columns=cols, index=False)


# --- Main ---


def main(args):
    wt_seqs = pd.read_csv(args.reference_file)

    # Filter by RNA type
    is_non_coding = wt_seqs["RNA_TYPE"].str.contains(
        "ribozyme|tRNA|aptamer|splicing", case=False, regex=True
    )
    # ncRNA is the stricter set used by the fitness leaderboard: it drops the
    # mRNA-splicing assays, whose readout is splicing efficiency rather than the
    # fitness of a non-coding RNA.
    is_ncrna = wt_seqs["RNA_TYPE"].str.contains(
        "ribozyme|tRNA|aptamer", case=False, regex=True
    )
    if args.type == "ncRNA":
        wt_seqs = wt_seqs[is_ncrna]
    elif args.type == "non-coding":
        wt_seqs = wt_seqs[is_non_coding]
    elif args.type == "coding":
        wt_seqs = wt_seqs[~is_non_coding]
    elif args.type != "all":
        raise ValueError(
            f"Expected 'all', 'ncRNA', 'non-coding', or 'coding' for --type "
            f"(got {args.type})"
        )

    # Filter by EVmutation availability
    if args.msa_only:
        datasets_to_drop = []
        for _, row in wt_seqs.iterrows():
            dataset = row["DMS_ID"]
            df_path = f"{args.combined_dir}/{dataset}.csv"
            if not os.path.exists(df_path):
                datasets_to_drop.append(dataset)
                continue
            df = pd.read_csv(df_path)
            if (
                "EVmutation_score" not in df.columns
                or df["EVmutation_score"].dropna().shape[0] < 10
            ):
                datasets_to_drop.append(dataset)
        if datasets_to_drop:
            wt_seqs = wt_seqs[~wt_seqs["DMS_ID"].isin(datasets_to_drop)]
            print(
                f"Dropped {len(datasets_to_drop)} datasets missing EVmutation scores."
            )

    # Build model list
    model_list = [
        "evo1",
        "evo1.5",
        "evo2",
        "evo2_40b",
        "GenSLM",
        "NT",
        "rinalmo",
        "RNAErnie",
        "RNA-FM",
        "orthrus",
        "aido_rna",
        "rnagenesis",
        # AIDO.RNA size series, dropped automatically when not merged in.
        "aido_rna_1m",
        "aido_rna_25m",
        "aido_rna_300m",
        "aido_rna_650m",
    ]
    if args.msa_only:
        model_list.append("EVmutation")
    # Keep only models that were actually merged in. A model whose predictions are
    # absent would otherwise contribute an all-NaN row to every output table.
    model_list = filter_available_models(model_list, wt_seqs, args.combined_dir)
    score_columns = [f"{m}_score" for m in model_list]
    # Only aggregate over RNA types that survive the --type filter, otherwise the
    # absent types contribute NaN to every model's All_Mean.
    all_types = ["mRNA-splicing", "mRNA-coding", "tRNA", "Aptamer", "Ribozyme"]
    types = [t for t in all_types if (wt_seqs["RNA_TYPE"] == t).any()]

    # Load data and compute all per-assay metrics in one pass
    wt_seqs = load_assay_metrics(
        wt_seqs, args.combined_dir, score_columns, args.msa_only
    )

    # Aggregate
    result_by_type = aggregate_by_rna_type(
        wt_seqs, types, score_columns, args.calculate_se
    )
    result_by_depth = aggregate_by_depth(wt_seqs, score_columns)
    result_by_type_and_depth = aggregate_by_type_and_depth(
        wt_seqs, types, score_columns
    )

    # Save all outputs
    os.makedirs(args.performance_dir, exist_ok=True)
    result_by_type.to_csv(
        os.path.join(args.performance_dir, "results_by_rna_type.csv"), index=False
    )
    result_by_depth.to_csv(
        os.path.join(args.performance_dir, "results_by_mutation_depth.csv"), index=False
    )
    result_by_type_and_depth.to_csv(
        os.path.join(args.performance_dir, "results_by_rna_type_and_depth.csv"),
        index=False,
    )
    save_assay_level_results(
        wt_seqs,
        score_columns,
        os.path.join(args.performance_dir, "assay_level_results.csv"),
    )
    save_assay_level_results_transposed(
        wt_seqs,
        score_columns,
        os.path.join(args.performance_dir, "assay_level_results_transposed.csv"),
    )

    print("\nMetrics by RNA Type:")
    print(result_by_type)
    print("\nMetrics by Mutation Depth:")
    print(result_by_depth)
    print("\nMetrics by RNA Type and Mutation Depth:")
    print(result_by_type_and_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze RNA datasets and calculate metrics for multiple models."
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default="reference_sheet_final.csv",
        help="Path to the reference CSV file",
    )
    parser.add_argument(
        "--combined_dir",
        type=str,
        default="merged",
        help="Base directory for combined result files",
    )
    parser.add_argument(
        "--calculate_se",
        action="store_true",
        help="Calculate standard errors (computationally intensive)",
    )
    parser.add_argument(
        "--performance_dir",
        type=str,
        default=None,
        help="Dir where performance file should be stored",
    )
    parser.add_argument(
        "--msa_only",
        action="store_true",
        help="Only score models on assays where EVmutation has scores",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        choices=["all", "ncRNA", "non-coding", "coding"],
        help="Filter assays by type ('ncRNA' is ribozyme, tRNA and aptamer only)",
    )
    args = parser.parse_args()
    if args.performance_dir is None:
        args.performance_dir = os.getcwd()
    main(args)
