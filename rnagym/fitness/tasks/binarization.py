#!/usr/bin/env python3
"""Select per-assay DMS fitness binarization cutoffs (`bimodal_valley` vs `median`)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import argrelmin
from sklearn.mixture import GaussianMixture

from rnagym.config import ConfigFitness
from rnagym.fitness.data import read_assay, read_reference, write_csv_atomically
from rnagym.fitness.tasks.model_registry import ASSAY_GROUPS

BIN_CUTOFF = "DMS_bin_cutoff"
BIN_METHOD = "DMS_bin_method"
FRAC_FUNCTIONAL = "DMS_frac_functional"
MIN_CLASS_FRACTION = 0.05
MIN_CLASS_VARIANTS = 25


def fit_gmm_valley(values: np.ndarray) -> tuple[float, float | None, float] | None:
    """Fit a regularized 2-component 1D GMM and locate any interior density valley.

    Returns ``(ashman_d, valley_position, valley_to_peak_ratio)``.
    """
    variance = float(np.var(values))
    if variance <= 0:
        return None
    gmm = GaussianMixture(
        n_components=2,
        reg_covar=max(1e-6, 0.01 * variance),
        random_state=0,
        n_init=10,
    ).fit(values[:, None])
    means = np.asarray(gmm.means_).ravel()
    variances = np.asarray(gmm.covariances_).ravel()
    order = np.argsort(means)
    m0, m1 = float(means[order[0]]), float(means[order[1]])
    s0, s1 = float(np.sqrt(variances[order[0]])), float(np.sqrt(variances[order[1]]))
    ashman_d = abs(m1 - m0) / float(np.sqrt(0.5 * (s0**2 + s1**2)))
    grid = np.linspace(m0, m1, 4000)
    density = np.exp(gmm.score_samples(grid[:, None]))
    minima = argrelmin(density)[0]
    if len(minima) == 0:
        return ashman_d, None, 1.0
    idx = int(minima[int(np.argmin(density[minima]))])
    valley_ratio = float(density[idx] / min(density[0], density[-1]))
    return ashman_d, float(grid[idx]), valley_ratio


def otsu_threshold(values: np.ndarray, nbins: int = 512) -> float:
    """Maximize between-class variance across a 1D histogram."""
    counts, edges = np.histogram(values, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(counts)[:-1]
    w1 = counts.sum() - w0
    valid = (w0 > 0) & (w1 > 0)
    weighted = np.cumsum(counts * centers)[:-1]
    mu0 = weighted[valid] / w0[valid]
    mu1 = (float(np.sum(counts * centers)) - weighted[valid]) / w1[valid]
    between = (w0[valid] / counts.sum()) * (w1[valid] / counts.sum()) * (mu0 - mu1) ** 2
    return float(centers[:-1][valid][int(np.argmax(between))])


def parse_wt_score(raw_value: object) -> float | None:
    """Parse numeric wild-type DMS score from reference sheet annotations."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if "In fitness score:" in text:
        return float(text.split("In fitness score:")[1].strip())
    if text.startswith("Kd:"):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def select_binarization_cutoff(
    scores: np.ndarray, wt_score: float | None = None
) -> tuple[float, str]:
    """Choose a per-assay binarization cutoff (`DMS_score > cutoff` is functional).

    Selects ``bimodal_valley`` between inactive and functional modes when a
    well-separated valley exists, wild-type fitness lies above the cutoff, and
    both classes have at least 5% support and 25 variants; otherwise ``median``.
    """
    median = float(np.median(scores))
    unique_vals, counts = np.unique(scores, return_counts=True)
    mode_idx = int(np.argmax(counts))
    mode_val = float(unique_vals[mode_idx])
    mode_frac = float(counts[mode_idx] / len(scores))

    def valid_split(cutoff: float) -> bool:
        positives = int((scores > cutoff).sum())
        negatives = len(scores) - positives
        wt_ok = True if wt_score is None else (wt_score > cutoff)
        return (
            wt_ok
            and min(positives, negatives) / len(scores) >= MIN_CLASS_FRACTION
            and min(positives, negatives) >= MIN_CLASS_VARIANTS
        )

    # Majority left-censored floor (>=50% tied at exact minimum): check whether
    # uncensored variants still separate impaired basal growth from WT-like fitness
    if mode_frac >= 0.50 and mode_val == float(scores.min()):
        tail = scores[scores > mode_val]
        tail_fit = fit_gmm_valley(tail) if len(tail) >= 2 * MIN_CLASS_VARIANTS else None
        if tail_fit is not None and tail_fit[0] >= 2.0:
            otsu = otsu_threshold(scores)
            if valid_split(otsu):
                return round(otsu, 6), "bimodal_valley"
        return median, "median"

    # Ignore discrete lower-boundary clamping spikes on signed log-fitness scales
    fit_scores = scores
    span = float(scores.max() - scores.min())
    if (
        span > 0
        and mode_frac >= 0.05
        and (mode_val - float(scores.min())) <= 0.06 * span
        and float(scores.min()) < 0
    ):
        fit_scores = scores[scores != mode_val]

    linear_fit = fit_gmm_valley(fit_scores)
    if linear_fit is not None and linear_fit[1] is not None:
        ashman_d, valley, ratio = linear_fit[0], linear_fit[1], linear_fit[2]
        if (
            (ashman_d >= 1.80 and ratio <= 0.78) or (ashman_d >= 2.50 and ratio <= 0.89)
        ) and valid_split(valley):
            return round(valley, 6), "bimodal_valley"

    if float(scores.min()) > 0:
        log_fit = fit_gmm_valley(np.log10(scores))
        if log_fit is not None and log_fit[1] is not None:
            ashman_d, log_valley, ratio = log_fit[0], log_fit[1], log_fit[2]
            raw_valley = float(10.0**log_valley)
            if ashman_d >= 1.60 and ratio <= 0.78 and valid_split(raw_valley):
                return round(raw_valley, 6), "bimodal_valley"

    return median, "median"


def update_reference_cutoffs(reference_path: Path, assay_dir: Path) -> pl.DataFrame:
    """Populate binarization columns on ncRNA rows of ``reference_sheet_final.csv``."""
    reference = read_reference(reference_path)
    ncrna_types = set(ASSAY_GROUPS["ncRNA"])
    cutoffs: list[float | None] = []
    methods: list[str | None] = []
    fractions: list[float | None] = []

    for row in reference.iter_rows(named=True):
        if row["RNA_TYPE"] not in ncrna_types:
            cutoffs.append(None)
            methods.append(None)
            fractions.append(None)
            continue
        assay = read_assay(assay_dir / f"{row['DMS_ID']}.csv").drop_nulls("mutant")
        scores = assay["DMS_score"].cast(pl.Float64).to_numpy()
        wt_score = parse_wt_score(row.get("WT_DMS_SCORE"))
        cutoff, method = select_binarization_cutoff(scores, wt_score)
        frac = round(float((scores > cutoff).mean()), 6)
        cutoffs.append(cutoff)
        methods.append(method)
        fractions.append(frac)

    updated = reference.with_columns(
        pl.Series(BIN_CUTOFF, cutoffs, dtype=pl.Float64),
        pl.Series(BIN_METHOD, methods, dtype=pl.String),
        pl.Series(FRAC_FUNCTIONAL, fractions, dtype=pl.Float64),
    )
    write_csv_atomically(updated, reference_path)
    return updated


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point to refresh ncRNA binarization cutoffs in the reference sheet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ConfigFitness.REFERENCE_FILE)
    parser.add_argument("--assays", type=Path, default=ConfigFitness.COMBINED_DIR)
    args = parser.parse_args(argv)
    updated = update_reference_cutoffs(args.reference, args.assays)
    ncrna = updated.filter(pl.col("RNA_TYPE").is_in(ASSAY_GROUPS["ncRNA"]))
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(
            ncrna.select("DMS_ID", "RNA_TYPE", BIN_CUTOFF, BIN_METHOD, FRAC_FUNCTIONAL)
        )


if __name__ == "__main__":
    main()
