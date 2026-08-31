#!/usr/bin/env python3
"""
Compare the masked-marginal fill strategies across checkpoints.

Reads prediction files written by ``rnagym.fitness.baselines.masked_lm``, one
per assay per checkpoint, each holding every computed strategy as its own column, and
reports the benchmark metric: signed Spearman per assay, averaged within each
ncRNA category, then a macro mean over the categories so that the 26 ribozyme
assays do not swamp the 3 tRNA and 2 aptamer assays.

Also reports where the strategies differ, whether they reorder the checkpoints,
and how the comparison responds to dropping the category weighting. This is the
code behind the sensitivity section of leaderboard/fitness/README.md.

The folders and columns come from model_registry.SCORE_COLS, so the
registry is not duplicated here.

Example:
    python -m rnagym.fitness.tasks.analyze_fill_strategies
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from rnagym.config import ConfigFitness
from rnagym.fitness.tasks.model_registry import (
    FOUR_FILL_MODELS,
    SCORE_COLS,
    resolve_source,
)

NCRNA = ["Ribozyme", "tRNA", "Aptamer"]
STRATEGIES = ["wt_fill", "mask_fill", "mut_fill", "match_fill"]


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions_folder",
        type=Path,
        default=ConfigFitness.PREDICTION_DIR,
        help="Folder holding one subfolder per prediction set",
    )
    parser.add_argument(
        "--ref_sheet",
        type=Path,
        default=ConfigFitness.REFERENCE_FILE,
        help="Reference sheet with DMS_ID and RNA_TYPE",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=None,
        help="Where to write the per-assay and macro CSVs (default: no CSVs)",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Report on checkpoints missing some assays instead of skipping them",
    )
    return parser.parse_args(argv)


def checkpoints():
    """Group the registry's four-fill entries by checkpoint."""
    grouped = {}
    for entry in FOUR_FILL_MODELS:
        for strategy in STRATEGIES:
            suffix = f"_{strategy}"
            if entry.endswith(suffix):
                grouped.setdefault(entry[: -len(suffix)], {})[strategy] = entry
                break
    incomplete = {
        name: sorted(set(STRATEGIES) - set(entries))
        for name, entries in grouped.items()
        if set(entries) != set(STRATEGIES)
    }
    if incomplete:
        raise ValueError(f"Incomplete four-fill registry entries: {incomplete}")
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
        directory = Path(args.predictions_folder) / folders.pop()
        if not directory.is_dir():
            continue
        found = sorted(
            path.stem for path in directory.glob("*.csv") if path.stem in expected
        )
        if set(found) != set(expected):
            missing = sorted(set(expected) - set(found))
            print(
                f"{name:18s} incomplete: {len(found)}/{len(expected)} assays, "
                f"missing {missing[:4]}"
            )
            if not args.allow_incomplete:
                continue
        for assay in found:
            assay_path = directory / f"{assay}.csv"
            frame = pd.read_csv(assay_path)
            absent = [s for s, c in columns.items() if c not in frame.columns]
            if absent:
                raise KeyError(f"{assay_path} has no columns for {absent}")
            # Every strategy must cover the same variants, or the comparison
            # between them is confounded by coverage rather than by method
            values = {
                strategy: pd.to_numeric(frame[column], errors="raise").to_numpy()
                for strategy, column in columns.items()
            }
            infinite = [
                strategy
                for strategy, strategy_values in values.items()
                if np.isinf(strategy_values).any()
            ]
            if infinite:
                raise ValueError(f"{assay_path} has infinite scores for {infinite}")
            masks = {
                strategy: np.isfinite(strategy_values)
                for strategy, strategy_values in values.items()
            }
            if not all(
                np.array_equal(mask, masks["wt_fill"])
                for strategy, mask in masks.items()
                if strategy != "wt_fill"
            ):
                raise ValueError(f"{assay_path}: strategies cover different variants")
            dms_score = pd.to_numeric(frame["DMS_score"], errors="raise").to_numpy()
            if np.isinf(dms_score).any():
                raise ValueError(f"{assay_path} has infinite DMS scores")
            for strategy in STRATEGIES:
                usable = np.isfinite(dms_score) & masks[strategy]
                if usable.sum() < 3:
                    raise ValueError(f"{assay}: only {usable.sum()} usable variants")
                rho = stats.spearmanr(
                    dms_score[usable], values[strategy][usable]
                ).correlation
                if not np.isfinite(rho):
                    raise ValueError(f"{assay}: Spearman undefined for {strategy}")
                records.append(
                    {
                        "model": name,
                        "strategy": strategy,
                        "assay": assay,
                        "RNA_TYPE": rna_type[assay],
                        "n": int(usable.sum()),
                        "spearman": rho,
                    }
                )
    return pd.DataFrame(records)


def macro_table(per_assay):
    rows = []
    for (model, strategy), group in per_assay.groupby(["model", "strategy"]):
        by_type = {t: group[group.RNA_TYPE == t]["spearman"].mean() for t in NCRNA}
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                **by_type,
                "macro_3ncRNA": float(np.mean([by_type[t] for t in NCRNA])),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None):
    """Compare masked-marginal fill strategies."""
    args = parse_args(argv)
    ref = pd.read_csv(args.ref_sheet, encoding="utf-8-sig")
    required = {"DMS_ID", "RNA_TYPE"}
    missing = sorted(required - set(ref.columns))
    if missing:
        raise ValueError(f"Reference sheet is missing columns: {missing}")
    if ref[list(required)].isna().any().any():
        raise ValueError("Reference sheet has missing DMS_ID or RNA_TYPE values")
    duplicated = sorted(ref.loc[ref["DMS_ID"].duplicated(), "DMS_ID"].astype(str))
    if duplicated:
        raise ValueError(f"Reference sheet repeats DMS_ID values: {duplicated[:5]}")
    rna_type = dict(zip(ref["DMS_ID"], ref["RNA_TYPE"]))
    expected = sorted(d for d, t in rna_type.items() if t in NCRNA)
    print(f"{len(expected)} ncRNA assays expected per checkpoint")

    per_assay = collect(args, expected, rna_type)
    if per_assay.empty:
        raise SystemExit("No complete checkpoint found in the predictions folder")
    macro = macro_table(per_assay)
    wide = macro.pivot(index="model", columns="strategy", values="macro_3ncRNA")[
        STRATEGIES
    ]
    order = list(wide.sort_values("wt_fill", ascending=False).index)

    print("\n=== signed Spearman, macro over the 3 ncRNA categories ===")
    print(wide.loc[order].round(4).to_string())

    print("\n=== ordering under each strategy, best first ===")
    for strategy in STRATEGIES:
        print(
            f"  {strategy:10s} "
            + " > ".join(wide[strategy].sort_values(ascending=False).index)
        )

    print("\n=== where the strategies differ ===")
    category_scores = macro.set_index(["model", "strategy"])[NCRNA]
    by_model = category_scores.groupby(level="model")
    spread = (by_model.max() - by_model.min()).loc[order]
    print(spread.round(4).to_string())
    print("  mean spread: " + ", ".join(f"{c} {spread[c].mean():.4f}" for c in NCRNA))

    print("\n=== without the category weighting ===")
    flat = (
        per_assay.groupby(["model", "strategy"])["spearman"]
        .mean()
        .unstack()[STRATEGIES]
    )
    print(flat.loc[order].round(4).to_string())
    print(
        f"  wt-fill best on {(wide.loc[order].idxmax(axis=1) == 'wt_fill').sum()}/{len(order)} "
        f"under the macro metric, {(flat.loc[order].idxmax(axis=1) == 'wt_fill').sum()}/{len(order)} "
        "under a flat mean over all assays"
    )

    if args.output_folder:
        out = Path(args.output_folder)
        out.mkdir(parents=True, exist_ok=True)
        per_assay.to_csv(out / "fill_strategy_per_assay.csv", index=False)
        macro.to_csv(out / "fill_strategy_macro.csv", index=False)
        print(
            f"\nwrote {out}/fill_strategy_per_assay.csv and {out}/fill_strategy_macro.csv"
        )


if __name__ == "__main__":
    main()
