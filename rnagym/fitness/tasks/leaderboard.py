"""Build the fitness leaderboard and comparison on EVmutation-covered variants."""

from __future__ import annotations

import argparse
import shlex
import tempfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from rnagym.config import ConfigFitness
from rnagym.fitness.tasks import performance_fitness

CATEGORIES = ("Ribozyme", "tRNA", "Aptamer")
MODEL_LABELS = {
    "aido_rna": "AIDO.RNA (1.6B)",
    "aido_rna_1m": "AIDO.RNA (1M)",
    "aido_rna_25m": "AIDO.RNA (25M)",
    "aido_rna_300m": "AIDO.RNA (300M)",
    "aido_rna_650m": "AIDO.RNA (650M)",
    "evo1": "Evo 1",
    "evo1.5": "Evo 1.5",
    "evo2": "Evo 2 (7B)",
    "evo2_1b_base": "Evo 2 (1B base)",
    "evo2_20b": "Evo 2 (20B)",
    "evo2_40b": "Evo 2 (40B)",
    "EVmutation": "EVmutation",
    "GenSLM": "GenSLM (2.5B)",
    "ntv3_8m": "Nucleotide Transformer v3 (8M)",
    "ntv3_100m": "Nucleotide Transformer v3 (100M)",
    "ntv3_650m": "Nucleotide Transformer v3 (650M)",
    "orthrus": "Orthrus",
    "rinalmo": "RiNALMo",
    "RNAErnie": "RNA-ERNIE",
    "RNA-FM": "RNA-FM",
    "rnagenesis": "RNAGenesis",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Recompute both tables from merged predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ConfigFitness.COMBINED_DIR)
    parser.add_argument("--output", type=Path, default=ConfigFitness.LEADERBOARD_DIR)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="rnagym-fitness-") as directory:
        reports = Path(directory)
        performance_fitness.main(
            performance_fitness.parse_args(
                shlex.split(
                    f"--input {shlex.quote(str(args.input))} --output {shlex.quote(str(reports))}"
                )
            )
        )
        write_tables(reports, args.output)
    print(f"Wrote fitness tables to {args.output}")


def render_table(table: pl.DataFrame, coverage: pl.DataFrame, partial: bool) -> str:
    """Format category means, marking the baseline's partial coverage in the main table."""
    headers = [
        f"{name} (n={coverage.filter(pl.col('RNA_TYPE') == name).height})"
        for name in CATEGORIES
    ]
    lines = [
        "| Rank | Model | " + " | ".join(headers) + " | Macro (3 ncRNA) |",
        "|---:|:--|--:|--:|--:|--:|",
    ]
    for rank, row in enumerate(table.iter_rows(named=True), 1):
        model = MODEL_LABELS[row["model"]]
        if partial and row["model"] == "EVmutation":
            model += "*"
        values = [
            "-" if row[column] is None else f"{row[column]:.4f}"
            for column in (*CATEGORIES, "macro_3ncRNA")
        ]
        place = str(rank) if row["macro_3ncRNA"] is not None else "-"
        lines.append(f"| {place} | {model} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_tables(reports: Path, output: Path) -> None:
    """Write full precision scores, coverage counts and the README tables."""
    coverage = pl.read_csv(reports / "evmutation/coverage.csv")
    covered = coverage.filter(pl.col("evaluated_variants") > 0)
    variants = int(coverage["evaluated_variants"].sum())
    total = int(coverage["variants"].sum())
    blocks = []
    output.mkdir(parents=True, exist_ok=True)
    for directory, filename, subset in (
        (reports, "leaderboard_signed_3ncRNA.csv", coverage),
        (reports / "evmutation", "leaderboard_evmutation.csv", covered),
    ):
        table = (
            pl.read_csv(directory / "results_by_rna_type.csv")
            .select(
                pl.col("Model").str.strip_suffix("_score").alias("model"),
                *[pl.col(f"Spearman_{name}_Mean").alias(name) for name in CATEGORIES],
                pl.col("Spearman_All_Mean").alias("macro_3ncRNA"),
            )
            .sort("macro_3ncRNA", descending=True, nulls_last=True)
        )
        table.write_csv(output / filename)
        blocks.append(render_table(table, subset, partial=directory == reports))
    coverage.write_csv(output / "evmutation_coverage.csv")
    generated = (
        blocks[0]
        + f"\n\n\\* EVmutation scores {covered.height}/{coverage.height} assays and "
        + f"{variants:,}/{total:,} variants ({variants / total:.1%}). "
        + "Its row uses those variants. [Coverage by assay][5].\n\n"
        + "### EVmutation-covered variants\n\n"
        + f"All models use the same {variants:,} variants from {covered.height} assays. "
        + "[Full precision CSV][6].\n\n"
        + blocks[1]
    )
    readme = output / "README.md"
    start, end = "<!-- BEGIN GENERATED TABLES -->", "<!-- END GENERATED TABLES -->"
    text = (
        readme.read_text()
        if readme.exists()
        else f"# Fitness leaderboard\n\n{start}\n{end}\n\n[5]: evmutation_coverage.csv\n[6]: leaderboard_evmutation.csv\n"
    )
    before, rest = text.split(start)
    _, after = rest.split(end)
    readme.write_text(f"{before}{start}\n{generated}\n{end}{after}")


if __name__ == "__main__":
    main()
