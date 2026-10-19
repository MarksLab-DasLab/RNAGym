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
from rnagym.fitness.tasks.model_registry import BASELINE_MODELS

CATEGORIES = ("Ribozyme", "tRNA", "Aptamer")
# Mutation counts in the README tables; deeper counts rest on one or two molecules
TABLE_MUTATIONS = range(1, 10)
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
    "mutation_count": "Mutation count",
    "ntv3_8m": "Nucleotide Transformer v3 (8M)",
    "ntv3_100m": "Nucleotide Transformer v3 (100M)",
    "ntv3_650m": "Nucleotide Transformer v3 (650M)",
    "orthrus": "Orthrus",
    "rinalmo": "RiNALMo",
    "RNA-2Di": "RNA-2Di",
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


def label(model: str) -> str:
    """Display name, with baselines in italics and EVmutation marked for the footnote."""
    name = MODEL_LABELS[model]
    if model in BASELINE_MODELS:
        name = f"_{name}_"
    if model == "EVmutation":
        name += "\\*"
    return name


def number(value: float | None) -> str:
    """Two decimals, printing a correlation that rounds to -0.0 as 0.00."""
    return "-" if value is None else f"{round(value, 2) + 0.0:.2f}"


def render_count_tables(counts: pl.DataFrame, order: list[str]) -> str:
    """Format Spearman within each mutation count, one table per RNA category.

    Mutation count is constant within a count, so it scores zero and is left out.
    """
    sections = []
    for name in CATEGORIES:
        category = counts.filter(
            (pl.col("RNA_TYPE") == name)
            & pl.col("mutations").is_in(list(TABLE_MUTATIONS))
        )
        # Molecules at each count, from the models scored on every variant
        molecules = dict(
            category.filter(pl.col("model") != "EVmutation")
            .group_by("mutations")
            .agg(pl.col("molecules").max())
            .iter_rows()
        )
        columns = sorted(molecules)
        if not columns:
            sections.append(f"{name}: no mutation count has enough variants.")
            continue
        values = {
            (row["model"], row["mutations"]): row["spearman"]
            for row in category.iter_rows(named=True)
        }
        lines = [
            f"{name}, by mutation count (molecules):",
            "",
            "| Model | " + " | ".join(f"{m} ({molecules[m]})" for m in columns) + " |",
            "|:--|" + "--:|" * len(columns),
        ]
        for model in order:
            if model == "mutation_count":
                continue
            cells = [number(values.get((model, m))) for m in columns]
            lines.append(f"| {label(model)} | " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def render_table(table: pl.DataFrame, coverage: pl.DataFrame) -> str:
    """Format category means to two decimals, with baselines in italics.

    EVmutation's partial coverage is marked for the footnote.
    """
    headers = []
    for name in CATEGORIES:
        assays = coverage.filter(pl.col("RNA_TYPE") == name)
        molecules = assays["MOLECULE_ID"].n_unique()
        headers.append(f"{name} ({assays.height} assays, {molecules} molecules)")
    lines = [
        "| Rank | Model | " + " | ".join(headers) + " | Macro (3 ncRNA) |",
        "|---:|:--|--:|--:|--:|--:|",
    ]
    for rank, row in enumerate(table.iter_rows(named=True), 1):
        values = [number(row[column]) for column in (*CATEGORIES, "macro_3ncRNA")]
        place = str(rank) if row["macro_3ncRNA"] is not None else "-"
        lines.append(
            f"| {place} | {label(row['model'])} | " + " | ".join(values) + " |"
        )
    return "\n".join(lines)


def write_tables(reports: Path, output: Path) -> None:
    """Write full precision scores, coverage counts and the README tables."""
    coverage = pl.read_csv(reports / "evmutation/coverage.csv")
    covered = coverage.filter(pl.col("evaluated_variants") > 0)
    variants = int(coverage["evaluated_variants"].sum())
    total = int(coverage["variants"].sum())
    generated = ""
    # Rows of the main table, in its rank order
    order: list[str] = []
    output.mkdir(parents=True, exist_ok=True)
    for directory, filename in (
        (reports, "leaderboard_signed_3ncRNA.csv"),
        (reports / "evmutation", "leaderboard_evmutation.csv"),
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
        if directory == reports:
            generated = render_table(table, coverage)
            order = table["model"].to_list()
    coverage.write_csv(output / "evmutation_coverage.csv")
    repeated = (
        coverage.group_by("MOLECULE_ID", maintain_order=True)
        .agg("DMS_ID")
        .filter(pl.col("DMS_ID").list.len() > 1)["DMS_ID"]
        .to_list()
    )
    if repeated:
        pairs = ", ".join(f"{ids[0]} with {' and '.join(ids[1:])}" for ids in repeated)
        generated += (
            "\n\nAssays of the same RNA molecule are averaged before their category "
            + f"({pairs}), as ProteinGym averages the assays of one protein."
        )
    generated += (
        f"\n\n\\* EVmutation scores {covered.height}/{coverage.height} assays and "
        + f"{variants:,}/{total:,} variants ({variants / total:.1%}). "
        + "Its row uses those variants. [Coverage by assay][5]. "
        + "[Compare all models on those variants][6]."
    )
    counts = pl.read_csv(reports / "results_by_rna_type_and_mutation_count.csv").select(
        pl.col("Model").str.strip_suffix("_score").alias("model"),
        "RNA_TYPE",
        pl.col("Mutations").alias("mutations"),
        pl.col("Spearman_Mean").alias("spearman"),
        pl.col("Molecules").alias("molecules"),
    )
    counts.write_csv(output / "leaderboard_by_mutation_count.csv")
    generated += (
        "\n\n### By mutation count\n\n"
        + "Spearman among variants with the same number of mutations within each "
        + "assay, for counts with at least 30 variants and varying measured fitness, "
        + "so the number of mutations alone ranks nothing. Assays of one molecule are "
        + "averaged first, then molecules within each category. Counts are never "
        + "averaged together. Molecule counts are those of the models scored on every "
        + "variant. EVmutation\\* uses only the variants it scores, so its cells can "
        + "rest on fewer molecules. Mutation count is left out, as it is constant "
        + "within a count. [All counts][8].\n\n"
        + render_count_tables(counts, order)
    )
    readme = output / "README.md"
    start, end = "<!-- BEGIN GENERATED TABLES -->", "<!-- END GENERATED TABLES -->"
    text = (
        readme.read_text()
        if readme.exists()
        else f"# Fitness leaderboard\n\n{start}\n{end}\n\n[5]: evmutation_coverage.csv\n[6]: leaderboard_evmutation.csv\n[8]: leaderboard_by_mutation_count.csv\n"
    )
    before, rest = text.split(start)
    _, after = rest.split(end)
    readme.write_text(f"{before}{start}\n{generated}\n{end}{after}")


if __name__ == "__main__":
    main()
