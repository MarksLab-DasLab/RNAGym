#!/usr/bin/env python3
"""
Compare the masked-marginal fill strategies across checkpoints.

Reads the prediction files written by fitness/baselines/masked_lm, one per assay
per checkpoint, each holding every computed strategy as its own column, and
reports the benchmark metric: signed Spearman per assay, averaged within each
ncRNA category, then a macro mean over the categories so that the 26 ribozyme
assays do not swamp the 3 tRNA and 2 aptamer assays.

Also reports where the strategies differ, whether they reorder the checkpoints,
how the comparison responds to dropping the category weighting, and a paired
bootstrap over assays. This is the code behind the sensitivity section of
leaderboard/fitness/README.md.

The folders and columns come from merge_scoring_files.SCORE_COLS, so the
registry is not duplicated here.

Example:
    python fitness/analyze_fill_strategies.py \\
        --predictions_folder path/to/model_predictions \\
        --ref_sheet fitness/reference_sheet_final.csv \\
        --output_folder leaderboard/fitness
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fitness.merge_scoring_files import (  # noqa: E402
    FOUR_FILL_MODELS,
    SCORE_COLS,
    resolve_source,
)

NCRNA = ["Ribozyme", "tRNA", "Aptamer"]
STRATEGIES = ["wt_fill", "mask_fill", "mut_fill", "match_fill"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions_folder", required=True,
                        help="Folder holding one subfolder per prediction set")
    parser.add_argument("--ref_sheet", required=True,
                        help="Reference sheet with DMS_ID and RNA_TYPE")
    parser.add_argument("--output_folder", default=None,
                        help="Where to write the per-assay and macro CSVs (default: no CSVs)")
    parser.add_argument("--bootstrap_draws", type=int, default=2000,
                        help="Paired bootstrap draws, 0 to skip (default: 2000)")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed (default: 0)")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Report on checkpoints missing some assays instead of skipping them")
    return parser.parse_args()


def checkpoints():
    """Group the registry's four-fill entries by checkpoint."""
    grouped = {}
    for entry in FOUR_FILL_MODELS:
        for strategy in STRATEGIES:
            suffix = f"_{strategy}"
            if entry.endswith(suffix):
                grouped.setdefault(entry[: -len(suffix)], {})[strategy] = entry
                break
    return grouped


def collect(args, expected, rna_type):
    """Per-assay signed Spearman for every checkpoint and strategy."""
    records = []
    for name, entries in checkpoints().items():
        columns, folders = {}, set()
        for strategy, entry in entries.items():
            folder, columns[strategy] = resolve_source(SCORE_COLS, entry)
            folders.add(folder)
        if len(folders) != 1:
            raise ValueError(f"{name}: its strategies are registered across {folders}")
        directory = os.path.join(args.predictions_folder, folders.pop())
        if not os.path.isdir(directory):
            continue
        found = sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".csv"))
        unexpected = sorted(set(found) - set(expected))
        if unexpected:
            raise ValueError(f"{name}: unexpected assays in {directory}: {unexpected}")
        if set(found) != set(expected):
            missing = sorted(set(expected) - set(found))
            print(f"{name:18s} incomplete: {len(found)}/{len(expected)} assays, "
                  f"missing {missing[:4]}")
            if not args.allow_incomplete:
                continue
        for assay in found:
            frame = pd.read_csv(os.path.join(directory, f"{assay}.csv"))
            absent = [s for s, c in columns.items() if c not in frame.columns]
            if absent:
                raise KeyError(f"{directory}/{assay}.csv has no columns for {absent}")
            # Every strategy must cover the same variants, or the comparison
            # between them is confounded by coverage rather than by method.
            masks = [frame[columns[s]].notna().to_numpy() for s in STRATEGIES]
            if not all((m == masks[0]).all() for m in masks[1:]):
                raise ValueError(f"{directory}/{assay}.csv: strategies cover different variants")
            for strategy in STRATEGIES:
                usable = frame[["DMS_score", columns[strategy]]].dropna()
                if len(usable) < 3:
                    raise ValueError(f"{assay}: only {len(usable)} usable variants")
                rho = stats.spearmanr(usable["DMS_score"], usable[columns[strategy]]).correlation
                if not np.isfinite(rho):
                    raise ValueError(f"{assay}: Spearman undefined for {strategy}")
                records.append({"model": name, "strategy": strategy, "assay": assay,
                                "RNA_TYPE": rna_type[assay], "n": len(usable), "spearman": rho})
    return pd.DataFrame(records)


def macro_table(per_assay):
    rows = []
    for (model, strategy), group in per_assay.groupby(["model", "strategy"]):
        by_type = {t: group[group.RNA_TYPE == t]["spearman"].mean() for t in NCRNA}
        rows.append({"model": model, "strategy": strategy, **by_type,
                     "macro_3ncRNA": float(np.mean([by_type[t] for t in NCRNA]))})
    return pd.DataFrame(rows)


def bootstrap(per_assay, order, draws, seed):
    """
    Paired bootstrap over assays, resampled within each category.

    The three categories are not resampled: they define the benchmark's estimand
    rather than sampling from a population. Each assay keeps its strategies
    together, so the comparison is paired.
    """
    rng = np.random.default_rng(seed)
    paired = per_assay.pivot_table(index=["model", "assay", "RNA_TYPE"],
                                   columns="strategy", values="spearman").reset_index()
    print(f"\n=== paired bootstrap, {draws} draws, seed {seed} ===")
    header = " ".join(f"{'wt-fill minus ' + s.replace('_', '-'):>26}"
                      for s in ["mut_fill", "mask_fill", "match_fill"])
    print(f"{'checkpoint':18s} {header}")
    for model in order:
        sub = paired[paired.model == model]
        by_type = {t: sub[sub.RNA_TYPE == t] for t in NCRNA}
        samples = {s: np.empty(draws) for s in STRATEGIES}
        for d in range(draws):
            drawn = {t: frame.iloc[rng.integers(0, len(frame), len(frame))]
                     for t, frame in by_type.items()}
            for strategy in STRATEGIES:
                samples[strategy][d] = np.mean([drawn[t][strategy].mean() for t in NCRNA])
        cells = []
        for other in ["mut_fill", "mask_fill", "match_fill"]:
            delta = samples["wt_fill"] - samples[other]
            lo, hi = np.percentile(delta, [2.5, 97.5])
            cells.append(f"{delta.mean():+.4f} [{lo:+.4f},{hi:+.4f}]" + ("*" if lo > 0 or hi < 0 else " "))
        print(f"{model:18s} " + " ".join(f"{c:>26}" for c in cells))
    print("  * = the 95% interval excludes zero. The tRNA and aptamer categories hold")
    print("  3 and 2 assays, so these intervals are wide by construction.")


def main():
    args = parse_args()
    ref = pd.read_csv(args.ref_sheet, encoding="utf-8-sig")
    rna_type = dict(zip(ref["DMS_ID"], ref["RNA_TYPE"]))
    expected = sorted(d for d, t in rna_type.items() if t in NCRNA)
    print(f"{len(expected)} ncRNA assays expected per checkpoint")

    per_assay = collect(args, expected, rna_type)
    if per_assay.empty:
        raise SystemExit("No complete checkpoint found in the predictions folder")
    macro = macro_table(per_assay)
    wide = macro.pivot(index="model", columns="strategy", values="macro_3ncRNA")[STRATEGIES]
    order = list(wide.sort_values("wt_fill", ascending=False).index)

    print("\n=== signed Spearman, macro over the 3 ncRNA categories ===")
    print(wide.loc[order].round(4).to_string())

    print("\n=== ordering under each strategy, best first ===")
    for strategy in STRATEGIES:
        print(f"  {strategy:10s} " + " > ".join(wide[strategy].sort_values(ascending=False).index))

    print("\n=== where the strategies differ ===")
    spread = pd.DataFrame([
        {"model": m, **{c: macro[macro.model == m].set_index("strategy")[c].pipe(
            lambda s: s.max() - s.min()) for c in NCRNA}} for m in order]).set_index("model")
    print(spread.round(4).to_string())
    print("  mean spread: " + ", ".join(f"{c} {spread[c].mean():.4f}" for c in NCRNA))

    print("\n=== without the category weighting ===")
    flat = per_assay.groupby(["model", "strategy"])["spearman"].mean().unstack()[STRATEGIES]
    print(flat.loc[order].round(4).to_string())
    print(f"  wt-fill best on {(wide.loc[order].idxmax(axis=1) == 'wt_fill').sum()}/{len(order)} "
          f"under the macro metric, {(flat.loc[order].idxmax(axis=1) == 'wt_fill').sum()}/{len(order)} "
          "under a flat mean over all assays")

    if args.bootstrap_draws:
        bootstrap(per_assay, order, args.bootstrap_draws, args.seed)

    if args.output_folder:
        out = Path(args.output_folder)
        out.mkdir(parents=True, exist_ok=True)
        per_assay.to_csv(out / "fill_strategy_per_assay.csv", index=False)
        macro.to_csv(out / "fill_strategy_macro.csv", index=False)
        print(f"\nwrote {out}/fill_strategy_per_assay.csv and {out}/fill_strategy_macro.csv")


if __name__ == "__main__":
    main()
