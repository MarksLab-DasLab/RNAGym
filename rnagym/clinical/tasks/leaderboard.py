"""Build the clinical leaderboard from the evaluation tables.

ClinVar has four units, pre-mRNA and mature mRNA SNVs and indels. A unit's
macro is the plain mean of the AUROC over its reportable consequence classes,
and the ClinVar score is the plain mean of the four unit macros, so each unit
weighs the same. MITOMAP is read on the processed RNA: the mito score is the
plain mean of the tRNA AUROC, on the tRNA with its CCA tail, and the mRNA
missense AUROC. Every model is compared on the rows all models scored. Every
aggregate uses the evaluation's full-precision values; tables are rounded to
12 digits only when written.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from rnagym.clinical.models import MODELS
from rnagym.config import ConfigClinical
from rnagym.fitness.tasks.leaderboard import MODEL_LABELS

# unit: (evaluated file, cohort)
NUCLEAR = {
    "premrna_snv": ("premrna_10101_snv", "snv"),
    "mature_snv": ("mature_mrna_snv", "snv"),
    "premrna_indel": ("premrna_10101_indel", "nonsnv"),
    "mature_indel": ("mature_mrna_indel", "nonsnv"),
}
UNIT_LABELS = {
    "premrna_snv": "Pre-mRNA SNV",
    "mature_snv": "Mature mRNA SNV",
    "premrna_indel": "Pre-mRNA indel",
    "mature_indel": "Mature mRNA indel",
}
MITO = "mito"
# Mito class: (evaluated file, label). The tRNA class is read on the tRNA with
# its CCA tail, which the mature tRNA carries
MITO_CLASSES = {
    "tRNA": ("mt_trna_cca", "tRNA + CCA"),
    "missense_variant": ("mt_rna", "mRNA missense"),
}
# Mito strata as named in the README note
STRATUM_LABELS = {
    "rRNA": "rRNA",
    "synonymous_variant": "synonymous",
    "lof_or_indel_coding": "loss of function",
    "other_coding": "other coding",
    "control_region": "control region",
    "noncanonical_or_intergenic": "intergenic",
}
# The mito file whose strata and rows describe the RNA view
MITO_VIEW = "mt_rna"
# Fitness label keys of models named differently there
FITNESS_KEYS = {"rna_fm": "RNA-FM", "evo2_7b": "evo2"}
FLOATS = ("auroc", "auroc_ci_low", "auroc_ci_high")
CLASS_COLUMNS = ("n_pathogenic", "n_benign", *FLOATS)


def label(model: str) -> str:
    """Return a model's display name, as on the fitness leaderboard."""
    return MODEL_LABELS[FITNESS_KEYS.get(model, model)]


def read_groups(evaluation: Path, name: str, models: Sequence[str]) -> pl.DataFrame:
    """Return one file's group table on the rows every model scored."""
    groups = pl.read_csv(
        evaluation / f"{name}.groups.csv",
        schema_overrides={c: pl.Float64 for c in FLOATS},
    ).filter((pl.col("row_set") == "model_rows") & pl.col("predictor").is_in(models))
    missing = set(models) - set(groups["predictor"].unique())
    if missing:
        raise ValueError(f"{name}: no evaluation for {sorted(missing)}")
    return groups


def check_complete(classes: pl.DataFrame, models: Sequence[str]) -> None:
    """Raise unless every class has every model, on the same rows, with an AUROC."""
    per_class = classes.group_by("unit", "class").agg(
        models=pl.col("model").n_unique(),
        rows=pl.len(),
        sizes=pl.struct("n_pathogenic", "n_benign").n_unique(),
    )
    bad = per_class.filter(
        (pl.col("models") != len(models))
        | (pl.col("rows") != len(models))
        | (pl.col("sizes") != 1)
    )
    if bad.height:
        raise ValueError(f"incomplete or inconsistent classes\n{bad}")
    if classes["auroc"].null_count():
        raise ValueError("a reportable class has no AUROC")


def nuclear(
    evaluation: Path, models: Sequence[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the reportable consequence classes and the macro of each ClinVar unit."""
    classes: list[pl.DataFrame] = []
    units: list[pl.DataFrame] = []
    for unit, (name, cohort) in NUCLEAR.items():
        groups = read_groups(evaluation, name, models).filter(
            (pl.col("cohort") == cohort)
            & (pl.col("group_type") == "consequence")
            & pl.col("reportable")
        )
        classes.append(
            groups.select(
                pl.lit(unit).alias("unit"),
                pl.col("group").alias("class"),
                pl.col("predictor").alias("model"),
                *CLASS_COLUMNS,
                pl.lit(name).alias("source"),
            )
        )
        headline = pl.read_csv(evaluation / f"{name}.headline.csv").filter(
            (pl.col("cohort") == cohort) & pl.col("predictor").is_in(models)
        )
        units.append(
            headline.select(
                pl.lit(unit).alias("unit"),
                pl.col("predictor").alias("model"),
                pl.col("consequence_macro_auroc").alias("macro"),
                pl.col("consequence_macro_ci_low").alias("ci_low"),
                pl.col("consequence_macro_ci_high").alias("ci_high"),
                pl.col("consequence_groups").alias("n_classes"),
                pl.col("rows_scored_by_all_models").alias("rows"),
            )
        )
    class_table, unit_table = pl.concat(classes), pl.concat(units)
    check_complete(class_table, models)
    # The evaluation's macro must be the plain mean of the classes kept here
    check = (
        class_table.group_by("unit", "model")
        .agg(mean=pl.col("auroc").mean(), k=pl.len())
        .join(unit_table, on=["unit", "model"], validate="1:1")
    )
    if check.height != len(NUCLEAR) * len(models):
        raise ValueError("unit macros and classes cover different models")
    off = check.filter(
        ((pl.col("mean") - pl.col("macro")).abs() > 1e-12)
        | (pl.col("k") != pl.col("n_classes"))
    )
    if off.height:
        raise ValueError(f"a unit macro is not the mean of its classes\n{off}")
    return class_table, unit_table


def mito_rows(groups: pl.DataFrame) -> pl.DataFrame:
    """Return the SNV groups of the one mito benign set."""
    return groups.filter(
        (pl.col("cohort") == "snv") & (pl.col("group_type") == "mito:all_benign")
    )


def mito(
    evaluation: Path, models: Sequence[str]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the mito classes, the mito macro and the RNA view's stratum counts."""
    parts = []
    for cls, (name, _) in MITO_CLASSES.items():
        kept = mito_rows(read_groups(evaluation, name, models)).filter(
            pl.col("group") == cls
        )
        if kept.height != len(models) or not kept["reportable"].all():
            raise ValueError(f"{name}: {cls} is missing or not reportable")
        parts.append(
            kept.select(
                pl.lit(MITO).alias("unit"),
                pl.col("group").alias("class"),
                pl.col("predictor").alias("model"),
                *CLASS_COLUMNS,
                pl.lit(name).alias("source"),
            )
        )
    classes = pl.concat(parts)
    check_complete(classes, models)
    view = mito_rows(read_groups(evaluation, MITO_VIEW, models))
    # The tRNA + CCA file must hold the same tRNA variants as the RNA view
    plain = view.filter(pl.col("group") == "tRNA").select("n_pathogenic", "n_benign")
    cca = classes.filter(pl.col("class") == "tRNA").select("n_pathogenic", "n_benign")
    if plain.unique().rows() != cca.unique().rows():
        raise ValueError("tRNA + CCA and mito RNA tRNA rows have different counts")
    everything = view.filter(pl.col("group") == "all")
    rows = (everything["n_pathogenic"] + everything["n_benign"]).unique()
    if rows.len() != 1:
        raise ValueError("models scored different mito rows")
    means = classes.group_by("model").agg(
        macro=pl.col("auroc").mean(), n_classes=pl.len()
    )
    units = means.select(
        pl.lit(MITO).alias("unit"),
        "model",
        "macro",
        pl.lit(None, pl.Float64).alias("ci_low"),
        pl.lit(None, pl.Float64).alias("ci_high"),
        pl.col("n_classes").cast(pl.Int64),
        pl.lit(rows.item(), pl.Int64).alias("rows"),
    )
    strata = view.filter(pl.col("group") != "all").select(
        "group", "n_pathogenic", "n_benign", "reportable"
    )
    sizes = strata.unique()
    if sizes["group"].n_unique() != sizes.height:
        raise ValueError("models disagree on mito stratum sizes")
    strata = sizes.select(
        pl.lit(MITO).alias("unit"),
        pl.col("group").alias("stratum"),
        "n_pathogenic",
        "n_benign",
        "reportable",
    ).sort("unit", "n_pathogenic", "n_benign", descending=[False, True, True])
    return classes, units, strata


def scores(units: pl.DataFrame, models: Sequence[str]) -> pl.DataFrame:
    """Return one row per model: the ClinVar and mito scores and the unit macros."""
    wide = units.pivot(on="unit", index="model", values="macro")
    names = [*NUCLEAR, MITO]
    if (
        wide.height != len(models)
        or wide.select(names).null_count().sum_horizontal().item()
    ):
        raise ValueError("a unit macro is missing")
    return wide.select(
        "model",
        pl.mean_horizontal(list(NUCLEAR)).alias("clinvar"),
        MITO,
        *NUCLEAR,
    ).sort("clinvar", "model", descending=[True, False])


def number(value: float) -> str:
    """Format an AUROC to two decimals."""
    return f"{value:.2f}"


def render(
    board: pl.DataFrame,
    units: pl.DataFrame,
    classes: pl.DataFrame,
    strata: pl.DataFrame,
) -> str:
    """Format the ClinVar and mito tables for the README."""
    unit_rows = units.group_by("unit").agg(
        pl.col("n_classes").first(), pl.col("rows").first()
    )
    sizes = {u: (k, n) for u, k, n in unit_rows.iter_rows()}
    headers = [
        f"{UNIT_LABELS[u]} ({sizes[u][0]} classes, {sizes[u][1]:,} variants)"
        for u in NUCLEAR
    ]
    lines = [
        "| Rank | Model | ClinVar | " + " | ".join(headers) + " |",
        "|---:|:--|--:|" + "--:|" * len(NUCLEAR),
    ]
    for rank, row in enumerate(board.iter_rows(named=True), 1):
        values = [number(row[u]) for u in ("clinvar", *NUCLEAR)]
        lines.append(f"| {rank} | {label(row['model'])} | " + " | ".join(values) + " |")
    clinvar = "\n".join(lines)

    counts = {
        c: (p, b)
        for c, p, b in classes.filter(pl.col("unit") == MITO)
        .select("class", "n_pathogenic", "n_benign")
        .unique()
        .iter_rows()
    }
    headers = [
        f"{text} ({counts[c][0]} P, {counts[c][1]} B)"
        for c, (_, text) in MITO_CLASSES.items()
    ]
    auroc = {
        (m, c): a
        for m, c, a in classes.filter(pl.col("unit") == MITO)
        .select("model", "class", "auroc")
        .iter_rows()
    }
    lines = [
        "| Rank | Model | Mito | " + " | ".join(headers) + " |",
        "|---:|:--|--:|" + "--:|" * len(MITO_CLASSES),
    ]
    order = board.sort(MITO, "model", descending=[True, False])
    for rank, row in enumerate(order.iter_rows(named=True), 1):
        values = [number(row[MITO])] + [
            number(auroc[row["model"], c]) for c in MITO_CLASSES
        ]
        lines.append(f"| {rank} | {label(row['model'])} | " + " | ".join(values) + " |")
    mito_table = "\n".join(lines)

    below = strata.filter(~pl.col("reportable"))
    notes = ", ".join(
        f"{STRATUM_LABELS.get(s, s)} ({p} P, {b} B)"
        for s, p, b in below.select("stratum", "n_pathogenic", "n_benign").iter_rows()
    )
    return (
        "### ClinVar\n\n"
        + "Macro AUROC within consequence classes, then the mean of the four units. "
        + "[95% intervals][1], [every class][2].\n\n"
        + clinvar
        + "\n\n### Mitochondrial (MITOMAP)\n\n"
        + "Mean of the tRNA and mRNA missense AUROC, on the processed RNA. "
        + "Strata with fewer than 10 pathogenic or 10 benign variants are left out "
        + f"of the mito score: {notes}.\n\n"
        + mito_table
    )


def write_tables(evaluation: Path, output: Path, models: Sequence[str]) -> None:
    """Write the leaderboard CSVs and the README tables."""
    nuclear_classes, nuclear_units = nuclear(evaluation, models)
    mito_classes, mito_units, strata = mito(evaluation, models)
    classes = pl.concat([nuclear_classes, mito_classes]).sort("unit", "class", "model")
    units = pl.concat([nuclear_units, mito_units]).sort("unit", "model")
    board = scores(units, models)
    output.mkdir(parents=True, exist_ok=True)
    board.write_csv(output / "leaderboard.csv", float_precision=12)
    units.write_csv(output / "units.csv", float_precision=12)
    classes.write_csv(output / "classes.csv", float_precision=12)
    strata.write_csv(output / "mito_strata.csv")
    generated = render(board, units, classes, strata)
    readme = output / "README.md"
    start, end = "<!-- BEGIN GENERATED TABLES -->", "<!-- END GENERATED TABLES -->"
    text = (
        readme.read_text()
        if readme.exists()
        else f"# Clinical leaderboard\n\n{start}\n{end}\n\n[1]: units.csv\n[2]: classes.csv\n"
    )
    before, rest = text.split(start)
    _, after = rest.split(end)
    readme.write_text(f"{before}{start}\n{generated}\n{end}{after}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=ConfigClinical.EVALUATION_DIR)
    parser.add_argument("--output", type=Path, default=ConfigClinical.LEADERBOARD_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Rebuild the leaderboard from an evaluation directory."""
    args = parse_args(argv)
    write_tables(args.input, args.output, args.models)
    print(f"Wrote clinical tables to {args.output}")


if __name__ == "__main__":
    main()
