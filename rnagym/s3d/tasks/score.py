"""Generate the RNAGym 3D leaderboard from released scores."""

import pandas as pd

from rnagym.config import Config3D

MODELS = {
    "AF3": "AlphaFold 3",
    "NU": "NuFold",
    "RF2NA": "RoseTTAFold2NA",
    "RHO": "RhoFold+",
    "TRRNA": "trRosettaRNA",
}
DATASETS = ("monomer", "multimer")
IDENTIFIERS = ["PDB ID", "Asym. Chain ID", "Auth. Chain ID"]


def score_dataset(dataset: str) -> pd.DataFrame:
    """Summarize every available model for one 3D dataset."""
    references = pd.read_parquet(Config3D.TARGET_FILE).query("type == @dataset")
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
        tm_train_column = f"{key} TM Homolog Score"
        if model_scores[tm_train_column].isna().any():
            raise RuntimeError("Run `pixi run usalign`, then `pixi run split`")
        completed = model_scores["tm_score"].notna().sum()
        inf_wc_samples = model_scores["inf_wc"].notna().sum()
        inf_nwc_samples = model_scores["inf_nwc"].notna().sum()
        model_scores = model_scores.assign(
            tm_score=model_scores["tm_score"].fillna(0),
            inf_wc=model_scores["inf_wc"].fillna(0),
            inf_nwc=model_scores["inf_nwc"].fillna(0),
        )
        units = model_scores.groupby(["cluster_rep", "sequence_id"]).agg(
            tm_score=("tm_score", "max"),
            tm_train=(tm_train_column, "max"),
            inf_wc=("inf_wc", "max"),
            inf_nwc=("inf_nwc", "max"),
        )
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

    result = pd.DataFrame(rows).sort_values("tm_score", ascending=False)
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
    Config3D.LEADERBOARD_README.write_text(f"{before}{start}\n{tables}\n{end}{after}")


def main() -> None:
    """Write detailed scores and the 3D leaderboard."""
    scores = pd.concat([score_dataset(dataset) for dataset in DATASETS])
    Config3D.LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(Config3D.LEADERBOARD_FILE, index=False)
    write_readme(scores)
    print(f"Wrote {Config3D.LEADERBOARD_FILE}")


if __name__ == "__main__":
    main()
