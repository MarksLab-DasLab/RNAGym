"""Score available predictions from every configured 3D model."""

import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr
from io import StringIO

import pandas as pd
from tqdm import tqdm

from rnagym.config import Config3D
from rnagym.s3d.util import Config
from rnagym.s3d.util.analysis import Analysis

IDENTIFIERS = ["PDB ID", "Asym. Chain ID", "Auth. Chain ID"]
SCORE_KEY = ["type", *IDENTIFIERS, "model"]


def prediction_available(target: dict, model: str) -> bool:
    """Check for a completed prediction without loading its reference structure."""
    baseline = Config.BASELINES[model]
    multimer = target["type"] == "multimer"
    name = target["PDB ID"] if multimer else target["sequence_id"]
    output_dir, prediction = Config.get_bl_out_pdb(model, name, multimer)
    return (
        baseline.name == "rho" or (output_dir / "SUCCESS").is_file()
    ) and prediction.is_file()


def score_target(job: tuple[dict, str]) -> dict:
    """Score one available model prediction against one target."""
    target, model = job
    baseline = Config.BASELINES[model]
    analysis = Analysis(
        target["PDB ID"],
        target["Asym. Chain ID"],
        target["Auth. Chain ID"],
        target["sequence_id"],
    )
    multimer = target["type"] == "multimer"
    tm_score = analysis.usa_result(baseline, multimer).tm_score
    with redirect_stderr(StringIO()):
        inf_wc, inf_nwc = analysis.inf(baseline, multimer)
    return {
        "type": target["type"],
        **{name: target[name] for name in IDENTIFIERS},
        "model": model.upper(),
        "tm_score": tm_score,
        "inf_wc": inf_wc,
        "inf_nwc": inf_nwc,
    }


def scoring_jobs(targets: list[dict]) -> list[tuple[dict, str]]:
    """Pair every target with each model that supports its target type."""
    return [
        (target, model)
        for model, baseline in Config.BASELINES.items()
        for target in targets
        if (target["type"] == "monomer" or baseline.mul_afa_file is not None)
        and prediction_available(target, model)
    ]


def main() -> None:
    """Score available predictions and update the shared score table."""
    targets = pd.read_parquet(Config3D.TARGET_FILE).to_dict("records")
    jobs = scoring_jobs(targets)
    if not jobs:
        print("No available predictions to score")
        return
    if not Config3D.MC_ANNOTATE.is_file():
        raise FileNotFoundError(f"Missing MC-Annotate: {Config3D.MC_ANNOTATE}")
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with ProcessPoolExecutor(workers) as pool:
        results = list(
            tqdm(
                pool.map(score_target, jobs),
                total=len(jobs),
                desc="Scoring predictions",
            )
        )

    new_scores = pd.DataFrame(results)
    previous = pd.read_parquet(Config3D.SCORE_FILE)
    scores = pd.concat([previous, new_scores], ignore_index=True)
    scores = scores.drop_duplicates(SCORE_KEY, keep="last").sort_values(SCORE_KEY)
    temporary = Config3D.SCORE_FILE.with_suffix(".tmp.parquet")
    scores.to_parquet(temporary, index=False)
    temporary.replace(Config3D.SCORE_FILE)
    print(f"Wrote {len(new_scores)} new scores to {Config3D.SCORE_FILE}")


if __name__ == "__main__":
    main()
