#!/usr/bin/env python3
"""Compare the four masked-marginal fill strategies across checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from scipy import stats

from rnagym.config import ConfigFitness
from rnagym.fitness.data import read_reference
from rnagym.fitness.tasks.model_registry import (
    ASSAY_GROUPS,
    FOUR_FILL_MODELS,
    SCORE_COLS,
    STRATEGIES,
    resolve_source,
)

NCRNA = ASSAY_GROUPS["ncRNA"]


def checkpoints():
    """Group the registry's four-fill entries by checkpoint."""
    grouped: dict[str, dict[str, str]] = {}
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


def collect(expected: Sequence[str], rna_type: dict[str, str]):
    """Per-assay signed Spearman for every checkpoint and strategy."""
    records = []
    for name, entries in checkpoints().items():
        columns: dict[str, str] = {}
        folders: set[str] = set()
        for strategy, entry in entries.items():
            folder, columns[strategy] = resolve_source(SCORE_COLS, entry)
            folders.add(folder)
        if len(folders) != 1:
            raise ValueError(f"{name}: its strategies are registered across {folders}")
        directory = ConfigFitness.PREDICTION_DIR / folders.pop()
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
            raise FileNotFoundError(f"{name}: missing assay predictions: {missing}")
        for assay in found:
            assay_path = directory / f"{assay}.csv"
            frame = pl.read_csv(assay_path)
            absent = [s for s, c in columns.items() if c not in frame.columns]
            if absent:
                raise KeyError(f"{assay_path} has no columns for {absent}")
            # Every strategy must cover the same variants, or the comparison
            # between them is confounded by coverage rather than by method
            values = {
                strategy: frame[column].cast(pl.Float64).to_numpy()
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
            dms_score = frame["DMS_score"].cast(pl.Float64).to_numpy()
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
    return pl.DataFrame(records)


def macro_table(per_assay: pl.DataFrame):
    rows = []
    for (model, strategy), group in per_assay.sort("model", "strategy").group_by(
        "model", "strategy", maintain_order=True
    ):
        by_type = {
            t: cast(
                float | None, group.filter(pl.col("RNA_TYPE") == t)["spearman"].mean()
            )
            for t in NCRNA
        }
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                **by_type,
                "macro_3ncRNA": float(
                    np.mean([value for value in by_type.values() if value is not None])
                )
                if all(value is not None for value in by_type.values())
                else None,
            }
        )
    return pl.DataFrame(rows)


def main(argv: Sequence[str] | None = None):
    """Compare masked-marginal fill strategies."""
    args = parse_args(argv)
    ref = read_reference(ConfigFitness.REFERENCE_FILE)
    rna_type = dict(ref.select("DMS_ID", "RNA_TYPE").iter_rows())
    expected = sorted(d for d, t in rna_type.items() if t in NCRNA)

    per_assay = collect(expected, rna_type)
    if per_assay.is_empty():
        raise SystemExit("No complete checkpoint found in the predictions folder")
    macro = macro_table(per_assay)
    wide = (
        macro.pivot(on="strategy", index="model", values="macro_3ncRNA")
        .select("model", *STRATEGIES)
        .sort("wt_fill", descending=True)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    per_assay.write_csv(args.output / "fill_strategy_per_assay.csv")
    macro.write_csv(args.output / "fill_strategy_macro.csv")
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(wide)
    print(f"Saved fill-strategy metrics to {args.output}")


def parse_args(argv: Sequence[str] | None = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ConfigFitness.REPORT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
