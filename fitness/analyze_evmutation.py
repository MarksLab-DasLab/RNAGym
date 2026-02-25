#!/usr/bin/env python3
"""EVmutation analysis report for RNAGym fitness benchmark.

Reads pre-computed results and merged data, prints a structured text report
covering coverage statistics, EVmutation performance, model comparison in the
alignment regime, and alignment vs. alignment-free regime analysis.

Usage:
    python fitness/analyze_evmutation.py --fitness_dir fitness/
"""

import argparse
import os

import numpy as np
import pandas as pd


def _fmt(val, decimals=3):
    """Format a float to fixed decimals, or return 'N/A'."""
    if pd.isna(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def _model_display(name):
    """Strip _score suffix for cleaner display."""
    return name.replace("_score", "")


def print_section(title, n=1):
    print(f"\n{'=' * 72}")
    print(f"  SECTION {n}: {title}")
    print(f"{'=' * 72}\n")


# ── Section 1 ────────────────────────────────────────────────────────────

def section_coverage(fitness_dir, ref):
    print_section("EVmutation Coverage Statistics", 1)

    noncoding = ref[ref["RNA_TYPE"] != "mRNA-coding"].copy()
    merged_dir = os.path.join(fitness_dir, "merged")

    rows = []
    for _, row in noncoding.iterrows():
        dms_id = row["DMS_ID"]
        rna_type = row["RNA_TYPE"]
        path = os.path.join(merged_dir, f"{dms_id}.csv")
        if not os.path.exists(path):
            rows.append((dms_id, rna_type, 0, 0, 0.0))
            continue
        df = pd.read_csv(path)
        total = len(df)
        if "EVmutation_score" in df.columns:
            scored = int(df["EVmutation_score"].notna().sum())
        else:
            scored = 0
        pct = 100.0 * scored / total if total > 0 else 0.0
        rows.append((dms_id, rna_type, total, scored, pct))

    cov = pd.DataFrame(rows, columns=["Assay", "RNA_TYPE", "Total", "EVmutation_scored", "Pct_coverage"])
    cov = cov.sort_values(["RNA_TYPE", "Assay"]).reset_index(drop=True)

    # Print table
    print(f"{'Assay':<40} {'RNA_TYPE':<15} {'Total':>7} {'EV_scored':>10} {'Cov%':>7}")
    print("-" * 82)
    for _, r in cov.iterrows():
        print(f"{r['Assay']:<40} {r['RNA_TYPE']:<15} {r['Total']:>7} {r['EVmutation_scored']:>10} {r['Pct_coverage']:>6.1f}%")

    usable = int((cov["EVmutation_scored"] >= 10).sum())
    total_assays = len(cov)
    total_dp = int(cov["Total"].sum())
    total_scored = int(cov["EVmutation_scored"].sum())
    overall_pct = 100.0 * total_scored / total_dp if total_dp > 0 else 0.0

    print()
    print(f"Summary: {usable}/{total_assays} non-coding assays usable (>= 10 scored datapoints)")
    print(f"         {total_scored:,}/{total_dp:,} total datapoints with EVmutation scores ({overall_pct:.1f}% coverage)")

    return cov


# ── Section 2 ────────────────────────────────────────────────────────────

def section_evmutation_perf(fitness_dir, ref):
    print_section("EVmutation Performance (Alignment Regime, 15 Assays)", 2)

    results_path = os.path.join(fitness_dir, "results_noncoding_msa_only", "assay_level_results.csv")
    results = pd.read_csv(results_path)

    ev = results[results["Model"] == "EVmutation_score"].copy()
    ev = ev.merge(ref[["DMS_ID", "RNA_TYPE"]], on="DMS_ID", how="left", suffixes=("", "_ref"))
    # Use RNA_TYPE from results if available, else from ref
    if "RNA_TYPE_ref" in ev.columns:
        ev["RNA_TYPE"] = ev["RNA_TYPE"].fillna(ev["RNA_TYPE_ref"])
        ev.drop(columns=["RNA_TYPE_ref"], inplace=True)
    ev = ev.sort_values("Spearman", ascending=False).reset_index(drop=True)

    print(f"{'Assay':<40} {'RNA_TYPE':<12} {'Spearman':>9} {'AUC':>7} {'MCC':>7}")
    print("-" * 78)
    for _, r in ev.iterrows():
        print(f"{r['DMS_ID']:<40} {r['RNA_TYPE']:<12} {_fmt(r['Spearman']):>9} {_fmt(r['AUC']):>7} {_fmt(r['MCC']):>7}")

    # Mean by RNA type
    print()
    print("Mean by RNA type:")
    grouped = ev.groupby("RNA_TYPE")[["Spearman", "AUC", "MCC"]].mean()
    for rna_type, row in grouped.iterrows():
        n = int((ev["RNA_TYPE"] == rna_type).sum())
        print(f"  {rna_type:<12} (n={n:>2}): Spearman={_fmt(row['Spearman'])}  AUC={_fmt(row['AUC'])}  MCC={_fmt(row['MCC'])}")

    overall = ev[["Spearman", "AUC", "MCC"]].mean()
    print(f"  {'Overall':<12} (n={len(ev):>2}): Spearman={_fmt(overall['Spearman'])}  AUC={_fmt(overall['AUC'])}  MCC={_fmt(overall['MCC'])}")

    # Best/worst
    best = ev.iloc[0]
    worst = ev.iloc[-1]
    print()
    print(f"Best:  {best['DMS_ID']} (Spearman={_fmt(best['Spearman'])})")
    print(f"Worst: {worst['DMS_ID']} (Spearman={_fmt(worst['Spearman'])})")

    return ev


# ── Section 3 ────────────────────────────────────────────────────────────

def section_model_comparison(fitness_dir, ref):
    print_section("Model Comparison on Alignment Regime (15 Assays, 9 Models)", 3)

    results_path = os.path.join(fitness_dir, "results_noncoding_msa_only", "assay_level_results.csv")
    results = pd.read_csv(results_path)

    # Pivot: rows = Model, columns = RNA_TYPE mean Spearman
    merged = results.merge(ref[["DMS_ID", "RNA_TYPE"]], on="DMS_ID", how="left", suffixes=("", "_ref"))
    if "RNA_TYPE_ref" in merged.columns:
        merged["RNA_TYPE"] = merged["RNA_TYPE"].fillna(merged["RNA_TYPE_ref"])
        merged.drop(columns=["RNA_TYPE_ref"], inplace=True)

    # Get unique RNA types present
    rna_types = sorted(merged["RNA_TYPE"].unique())

    # Mean Spearman per model per RNA type
    pivot = merged.pivot_table(values="Spearman", index="Model", columns="RNA_TYPE", aggfunc="mean")
    pivot["Overall"] = merged.groupby("Model")["Spearman"].mean()
    pivot = pivot.sort_values("Overall", ascending=False)

    # Print table
    header = f"{'Model':<20}"
    for rt in rna_types:
        n = merged[merged["RNA_TYPE"] == rt]["DMS_ID"].nunique()
        header += f" {rt}({n}):>10"
    # Redo with proper formatting
    col_headers = [f"{rt}({merged[merged['RNA_TYPE'] == rt]['DMS_ID'].nunique()})" for rt in rna_types]
    col_headers.append("Overall")

    header = f"{'Model':<20}"
    for ch in col_headers:
        header += f"  {ch:>12}"
    print(header)
    print("-" * (20 + 14 * len(col_headers)))

    for model, row in pivot.iterrows():
        line = f"{_model_display(model):<20}"
        for rt in rna_types:
            line += f"  {_fmt(row.get(rt, np.nan)):>12}"
        line += f"  {_fmt(row['Overall']):>12}"
        print(line)

    # Per-assay win counts
    print()
    print("Per-assay wins (which model has highest Spearman on each assay):")
    wins = {}
    for dms_id in merged["DMS_ID"].unique():
        sub = merged[merged["DMS_ID"] == dms_id]
        best_model = sub.loc[sub["Spearman"].idxmax(), "Model"]
        wins[best_model] = wins.get(best_model, 0) + 1

    for model, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {_model_display(model):<20} {count:>2} wins")

    print()
    print("Note: All scores computed on EVmutation-scoreable datapoints (inner join),")
    print("      making this a fair head-to-head comparison.")


# ── Section 4 ────────────────────────────────────────────────────────────

def section_alignment_regimes(fitness_dir, ref):
    print_section("Alignment vs. Alignment-Free Regime Comparison", 4)

    # Full results (8 foundation models, 33 non-coding assays, no EVmutation filtering)
    full_path = os.path.join(fitness_dir, "results_noncoding", "assay_level_results.csv")
    full = pd.read_csv(full_path)

    # MSA-only assay list (the 15 alignment-regime assays)
    msa_path = os.path.join(fitness_dir, "results_noncoding_msa_only", "assay_level_results.csv")
    msa = pd.read_csv(msa_path)
    alignment_assays = set(msa["DMS_ID"].unique())

    # Partition
    full["Regime"] = full["DMS_ID"].apply(
        lambda x: "Alignment" if x in alignment_assays else "Alignment-Free"
    )

    n_aln = full[full["Regime"] == "Alignment"]["DMS_ID"].nunique()
    n_free = full[full["Regime"] == "Alignment-Free"]["DMS_ID"].nunique()
    print(f"Alignment regime:      {n_aln} assays (EVmutation-scoreable)")
    print(f"Alignment-free regime: {n_free} assays (no usable EVmutation MSA)")
    print()

    # Mean Spearman per model per regime
    pivot = full.pivot_table(values="Spearman", index="Model", columns="Regime", aggfunc="mean")
    pivot["Delta"] = pivot.get("Alignment", 0) - pivot.get("Alignment-Free", 0)
    pivot = pivot.sort_values("Alignment", ascending=False)

    print(f"{'Model':<20} {'Alignment(15)':>14} {'Align-Free(18)':>15} {'Delta':>8}")
    print("-" * 60)
    for model, row in pivot.iterrows():
        aln = row.get("Alignment", np.nan)
        free = row.get("Alignment-Free", np.nan)
        delta = row.get("Delta", np.nan)
        sign = "+" if not pd.isna(delta) and delta > 0 else ""
        print(f"{_model_display(model):<20} {_fmt(aln):>14} {_fmt(free):>15} {sign}{_fmt(delta):>7}")

    # Note on regime
    print()
    print("Positive delta = model performs better on alignment-regime assays.")
    print("Negative delta = model performs better on alignment-free-regime assays.")

    # Which RNA types appear in each regime
    full_with_type = full.merge(ref[["DMS_ID", "RNA_TYPE"]], on="DMS_ID", how="left", suffixes=("", "_ref"))
    if "RNA_TYPE_ref" in full_with_type.columns:
        full_with_type["RNA_TYPE"] = full_with_type["RNA_TYPE"].fillna(full_with_type["RNA_TYPE_ref"])

    print()
    print("RNA types by regime:")
    for regime in ["Alignment", "Alignment-Free"]:
        sub = full_with_type[full_with_type["Regime"] == regime]
        types = sub.groupby("RNA_TYPE")["DMS_ID"].nunique().to_dict()
        types_str = ", ".join(f"{k}({v})" for k, v in sorted(types.items()))
        print(f"  {regime:<16}: {types_str}")


# ── Section 5 ────────────────────────────────────────────────────────────

def section_insights(fitness_dir, ref, cov_df):
    print_section("Insights", 5)

    # 1. Per-assay best model for alignment-regime assays
    msa_path = os.path.join(fitness_dir, "results_noncoding_msa_only", "assay_level_results.csv")
    msa = pd.read_csv(msa_path)

    print("(a) Per-assay best model (alignment regime, including EVmutation):")
    print()
    ev_wins = 0
    total = 0
    for dms_id in sorted(msa["DMS_ID"].unique()):
        sub = msa[msa["DMS_ID"] == dms_id]
        best_idx = sub["Spearman"].idxmax()
        best_model = sub.loc[best_idx, "Model"]
        best_sp = sub.loc[best_idx, "Spearman"]
        marker = " <-- EVmutation" if best_model == "EVmutation_score" else ""
        print(f"  {dms_id:<40} {_model_display(best_model):<15} Spearman={_fmt(best_sp)}{marker}")
        if best_model == "EVmutation_score":
            ev_wins += 1
        total += 1

    print()
    print(f"  EVmutation wins {ev_wins}/{total} assays ({100.0 * ev_wins / total:.0f}%)")

    # 2. Alignment-free-only RNA types / assays
    full_path = os.path.join(fitness_dir, "results_noncoding", "assay_level_results.csv")
    full = pd.read_csv(full_path)
    alignment_assays = set(msa["DMS_ID"].unique())
    free_assays = set(full["DMS_ID"].unique()) - alignment_assays

    full_with_type = full.merge(ref[["DMS_ID", "RNA_TYPE"]], on="DMS_ID", how="left", suffixes=("", "_ref"))
    if "RNA_TYPE_ref" in full_with_type.columns:
        full_with_type["RNA_TYPE"] = full_with_type["RNA_TYPE"].fillna(full_with_type["RNA_TYPE_ref"])

    free_types = full_with_type[full_with_type["DMS_ID"].isin(free_assays)]
    unique_free = free_types.groupby("RNA_TYPE")["DMS_ID"].apply(lambda x: sorted(x.unique())).to_dict()

    print()
    print("(b) Assays in alignment-free regime (no usable EVmutation MSA):")
    for rna_type, assays in sorted(unique_free.items()):
        print(f"  {rna_type}:")
        for a in assays:
            print(f"    - {a}")

    # 3. Coverage-performance correlation
    print()
    print("(c) Coverage vs. performance (EVmutation Spearman):")
    ev = msa[msa["Model"] == "EVmutation_score"][["DMS_ID", "Spearman"]].copy()
    ev = ev.merge(cov_df[["Assay", "Pct_coverage"]], left_on="DMS_ID", right_on="Assay", how="inner")

    if len(ev) >= 3:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(ev["Pct_coverage"], ev["Spearman"])
        print(f"  Spearman correlation between coverage % and EVmutation Spearman: {corr:.3f} (p={pval:.3f})")
        print()
        print(f"  {'Assay':<40} {'Cov%':>7} {'Spearman':>9}")
        print(f"  {'-'*58}")
        for _, r in ev.sort_values("Pct_coverage", ascending=False).iterrows():
            print(f"  {r['DMS_ID']:<40} {r['Pct_coverage']:>6.1f}% {_fmt(r['Spearman']):>9}")
    else:
        print("  Not enough data to compute correlation.")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EVmutation analysis report")
    parser.add_argument("--fitness_dir", required=True, help="Path to fitness/ directory")
    args = parser.parse_args()

    fitness_dir = args.fitness_dir

    print("=" * 72)
    print("  EVmutation Analysis Report — RNAGym Fitness Benchmark")
    print("=" * 72)

    ref = pd.read_csv(os.path.join(fitness_dir, "reference_sheet_final.csv"))

    cov_df = section_coverage(fitness_dir, ref)
    section_evmutation_perf(fitness_dir, ref)
    section_model_comparison(fitness_dir, ref)
    section_alignment_regimes(fitness_dir, ref)
    section_insights(fitness_dir, ref, cov_df)

    print()
    print("=" * 72)
    print("  End of Report")
    print("=" * 72)


if __name__ == "__main__":
    main()
