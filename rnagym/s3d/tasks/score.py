"""Generate the RNAGym 3D leaderboard from released scores."""

import polars as pl

from rnagym.config import Config3D
from rnagym.s3d.models import homology_columns

MODELS = {
    "AF3": "AlphaFold 3",
    "NU": "NuFold",
    "RF2NA": "RoseTTAFold2NA",
    "RHO": "RhoFold+",
    "TRRNA": "trRosettaRNA",
}
DATASETS = ("monomer", "multimer")
IDENTIFIERS = ["pdb_id", "asym_id", "auth_id"]
METRICS = ["tm_score", "inf_wc", "inf_nwc"]


def score_dataset(dataset: str) -> pl.DataFrame:
    """Summarize every available model for one 3D dataset.

    Parameters
    ----------
    dataset : str
        Target type, either ``monomer`` or ``multimer``.

    Returns
    -------
    pl.DataFrame
        Cluster-macro model scores ordered by TM score.
    """
    references = pl.read_parquet(Config3D.TARGET_FILE).filter(pl.col("type") == dataset)
    registry = pl.read_parquet(
        Config3D.SEQUENCE_FILE, columns=["sequence_id", "cluster_rep"]
    )
    references = references.join(registry, on="sequence_id", how="left", validate="m:1")
    if references["cluster_rep"].null_count():
        raise ValueError(f"Missing {dataset} sequence clusters")

    scores = pl.read_parquet(Config3D.SCORE_FILE).filter(pl.col("type") == dataset)
    if scores.select(*IDENTIFIERS, "model").is_duplicated().any():
        raise ValueError(f"Duplicate {dataset} score rows")

    rows = []
    for key, model in MODELS.items():
        predictions = scores.filter(pl.col("model") == key)
        if predictions.is_empty():
            continue
        model_scores = references.join(
            predictions, on=IDENTIFIERS, how="left", validate="1:1"
        )
        if model_scores.filter(
            pl.any_horizontal(pl.col("inf_wc", "inf_nwc") == -1)
        ).height:
            raise ValueError(f"Unresolved {dataset} INF score")
        tm_train_column = homology_columns(key)[-1]
        model_scores = model_scores.with_columns(
            pl.col(*METRICS, tm_train_column).fill_nan(None)
        )
        if model_scores[tm_train_column].null_count():
            raise RuntimeError("Run `pixi run usalign`, then `pixi run split`")
        value_columns = [*METRICS, tm_train_column]
        invalid = pl.any_horizontal(
            [
                pl.col(column).is_not_null()
                & ~pl.col(column).is_between(0, 1, closed="both")
                for column in value_columns
            ]
        )
        if model_scores.filter(invalid).height:
            raise ValueError(f"Invalid {dataset} score for {model}")
        completed = model_scores.height - model_scores["tm_score"].null_count()
        inf_wc_samples = model_scores.height - model_scores["inf_wc"].null_count()
        inf_nwc_samples = model_scores.height - model_scores["inf_nwc"].null_count()
        model_scores = model_scores.with_columns(
            pl.col(METRICS).fill_null(0),
            pl.col(tm_train_column).alias("tm_train"),
        )
        if dataset == "monomer":
            # One sequence prediction receives its best score across observed conformations
            units = model_scores.group_by("cluster_rep", "sequence_id").agg(
                pl.col(*METRICS, "tm_train").max()
            )
        else:
            # Each complex is a distinct prediction even when target sequences match
            units = model_scores.select(
                "cluster_rep", *IDENTIFIERS, *METRICS, "tm_train"
            )
        units = units.with_columns(
            (pl.col("tm_score") - pl.col("tm_train")).alias("delta_tm")
        )
        clusters = units.group_by("cluster_rep").agg(
            pl.col(*METRICS, "tm_train", "delta_tm").mean()
        )
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "samples": model_scores.height,
                "clusters": clusters.height,
                "completed": completed,
                "tm_score": clusters["tm_score"].mean(),
                "delta_tm": clusters["delta_tm"].mean(),
                "rho_tm": clusters.select(
                    pl.corr("tm_score", "tm_train", method="spearman")
                ).item(),
                "inf_wc": clusters["inf_wc"].mean(),
                "inf_wc_samples": inf_wc_samples,
                "inf_nwc": clusters["inf_nwc"].mean(),
                "inf_nwc_samples": inf_nwc_samples,
            }
        )

    return (
        pl.from_dicts(rows)
        .sort("tm_score", "model", descending=[True, False])
        .with_row_index("rank", offset=1)
    )


def markdown_table(scores: pl.DataFrame) -> str:
    """Format one dataset leaderboard as Markdown."""
    lines = [
        "| Rank | Model | TM | ΔTM | ρTM | INF-WC | INF-NWC |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scores.iter_rows(named=True):
        lines.append(
            f"| {row['rank']} | {row['model']} | {row['tm_score']:.3f} | "
            f"{row['delta_tm']:.3f} | {row['rho_tm']:.2f} | "
            f"{row['inf_wc']:.2f} | {row['inf_nwc']:.2f} |"
        )
    return "\n".join(lines)


def write_readme(scores: pl.DataFrame) -> None:
    """Update the generated tables in the leaderboard README."""
    sections = []
    for dataset, title in (("monomer", "Monomers"), ("multimer", "Multimers")):
        subset = scores.filter(pl.col("dataset") == dataset)
        n = subset.item(0, "samples")
        sections.append(f"### {title} (n={n} structures)\n\n{markdown_table(subset)}")

    start = "<!-- BEGIN GENERATED TABLES -->"
    end = "<!-- END GENERATED TABLES -->"
    readme = Config3D.LEADERBOARD_README.read_text()
    before, generated = readme.split(start)
    _, after = generated.split(end)
    tables = "\n\n".join(sections)
    temporary = Config3D.LEADERBOARD_README.with_suffix(".tmp")
    temporary.write_text(f"{before}{start}\n{tables}\n{end}{after}")
    temporary.replace(Config3D.LEADERBOARD_README)


def main() -> None:
    """Write detailed scores and the 3D leaderboard."""
    scores = pl.concat([score_dataset(dataset) for dataset in DATASETS])
    Config3D.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = Config3D.LEADERBOARD_FILE.with_suffix(".tmp")
    scores.write_csv(temporary)
    temporary.replace(Config3D.LEADERBOARD_FILE)
    write_readme(scores)
    print(f"Wrote {Config3D.LEADERBOARD_FILE}")


if __name__ == "__main__":
    main()
