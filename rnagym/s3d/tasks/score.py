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

    data = scores.merge(references, on=IDENTIFIERS, how="inner", validate="many_to_one")
    if len(data) != len(scores):
        raise ValueError(f"Missing {dataset} references")

    rows = []
    for key, model in MODELS.items():
        model_scores = data.query("model == @key")
        if model_scores.empty:
            continue
        if (model_scores[["inf_wc", "inf_nwc"]] == -1).any().any():
            raise ValueError(f"Unresolved {dataset} INF score")
        tm_train_column = f"{key} TM Homolog Score"
        tm_scores = model_scores["tm_score"].fillna(0)
        inf_wc = model_scores["inf_wc"].fillna(0)
        inf_nwc = model_scores["inf_nwc"].fillna(0)
        complete = pd.concat([tm_scores, model_scores[tm_train_column]], axis=1)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "samples": len(model_scores),
                "completed": model_scores["tm_score"].notna().sum(),
                "tm_score": tm_scores.mean(),
                "delta_tm": (complete["tm_score"] - complete[tm_train_column]).mean(),
                "rho_tm": complete.corr(method="spearman").iloc[0, 1],
                "inf_wc": inf_wc.mean(),
                "inf_wc_samples": inf_wc.notna().sum(),
                "inf_nwc": inf_nwc.mean(),
                "inf_nwc_samples": inf_nwc.notna().sum(),
            }
        )

    result = pd.DataFrame(rows).sort_values("tm_score", ascending=False)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def markdown_table(scores: pd.DataFrame) -> str:
    """Format one dataset leaderboard as Markdown."""
    lines = [
        "| Rank | Model | n | TM | ΔTM | ρTM | INF-WC | INF-NWC |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scores.itertuples():
        lines.append(
            f"| {row.rank} | {row.model} | {row.samples} | "
            f"{row.tm_score:.3f} | {row.delta_tm:.3f} | {row.rho_tm:.2f} | "
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
