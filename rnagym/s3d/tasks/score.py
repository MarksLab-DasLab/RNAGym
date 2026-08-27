"""Generate the RNAGym 3D leaderboard from released scores."""

import pandas as pd

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


def score_dataset(dataset: str) -> pd.DataFrame:
    """Summarize every available model for one 3D dataset."""
    references = pd.read_parquet(Config3D.TARGET_FILE).query("type == @dataset")
    registry = pd.read_parquet(
        Config3D.SEQUENCE_FILE, columns=["sequence_id", "cluster_rep"]
    )
    references = references.merge(
        registry, on="sequence_id", how="left", validate="many_to_one"
    )
    scores = pd.read_parquet(Config3D.SCORE_FILE).query("type == @dataset")
    if scores.duplicated([*IDENTIFIERS, "model"]).any():
        raise ValueError(f"Duplicate {dataset} score rows")

    rows = []
    for key, model in MODELS.items():
        predictions = scores.query("model == @key")
        if predictions.empty:
            continue
        model_scores = references.merge(
            predictions, on=IDENTIFIERS, how="left", validate="one_to_one"
        )
        if (model_scores[["inf_wc", "inf_nwc"]] == -1).any().any():
            raise ValueError(f"Unresolved {dataset} INF score")
        tm_train_column = homology_columns(key)[-1]
        if model_scores[tm_train_column].isna().any():
            raise RuntimeError("Run `pixi run usalign`, then `pixi run split`")
        observed = model_scores[[*METRICS, tm_train_column]].stack()
        if not observed.between(0, 1).all():
            raise ValueError(f"Invalid {dataset} score for {model}")
        completed = model_scores["tm_score"].notna().sum()
        inf_wc_samples = model_scores["inf_wc"].notna().sum()
        inf_nwc_samples = model_scores["inf_nwc"].notna().sum()
        model_scores = model_scores.assign(
            tm_score=model_scores["tm_score"].fillna(0),
            inf_wc=model_scores["inf_wc"].fillna(0),
            inf_nwc=model_scores["inf_nwc"].fillna(0),
        )
        model_scores = model_scores.rename(columns={tm_train_column: "tm_train"})
        if dataset == "monomer":
            # One sequence prediction receives its best score across observed conformations
            units = model_scores.groupby(["cluster_rep", "sequence_id"])[
                [*METRICS, "tm_train"]
            ].max()
        else:
            # Each complex is a distinct prediction even when target sequences match
            units = model_scores.set_index(["cluster_rep", *IDENTIFIERS])[
                [*METRICS, "tm_train"]
            ]
        units["delta_tm"] = units["tm_score"] - units["tm_train"]
        clusters = units.groupby("cluster_rep").mean()
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "samples": len(model_scores),
                "clusters": len(clusters),
                "completed": completed,
                "tm_score": clusters["tm_score"].mean(),
                "delta_tm": clusters["delta_tm"].mean(),
                "rho_tm": clusters[["tm_score", "tm_train"]]
                .corr(method="spearman")
                .iloc[0, 1],
                "inf_wc": clusters["inf_wc"].mean(),
                "inf_wc_samples": inf_wc_samples,
                "inf_nwc": clusters["inf_nwc"].mean(),
                "inf_nwc_samples": inf_nwc_samples,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["tm_score", "model"], ascending=[False, True]
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def markdown_table(scores: pd.DataFrame) -> str:
    """Format one dataset leaderboard as Markdown."""
    lines = [
        "| Rank | Model | TM | ΔTM | ρTM | INF-WC | INF-NWC |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scores.itertuples():
        lines.append(
            f"| {row.rank} | {row.model} | {row.tm_score:.3f} | "
            f"{row.delta_tm:.3f} | {row.rho_tm:.2f} | "
            f"{row.inf_wc:.2f} | {row.inf_nwc:.2f} |"
        )
    return "\n".join(lines)


def write_readme(scores: pd.DataFrame) -> None:
    """Update the generated tables in the leaderboard README."""
    sections = []
    for dataset, title in (("monomer", "Monomers"), ("multimer", "Multimers")):
        subset = scores[scores["dataset"] == dataset]
        sections.append(
            f"### {title} (n={subset['samples'].iloc[0]})\n\n{markdown_table(subset)}"
        )

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
    scores = pd.concat([score_dataset(dataset) for dataset in DATASETS])
    Config3D.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = Config3D.LEADERBOARD_FILE.with_suffix(".tmp")
    scores.to_csv(temporary, index=False)
    temporary.replace(Config3D.LEADERBOARD_FILE)
    write_readme(scores)
    print(f"Wrote {Config3D.LEADERBOARD_FILE}")


if __name__ == "__main__":
    main()
