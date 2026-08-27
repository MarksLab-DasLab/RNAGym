"""Score available predictions from every configured 3D model."""

import importlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr
from functools import cache
from io import StringIO
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rnagym.config import Config3D
from rnagym.s3d.util import Config
from rnagym.s3d.util.analysis import Analysis, valid_prediction

IDENTIFIERS = ["PDB ID", "Asym. Chain ID", "Auth. Chain ID"]
SCORE_KEY = ["type", *IDENTIFIERS, "model"]


@cache
def _completed_targets(model: str, kind: str) -> frozenset[str]:
    """Load targets with outputs current to their model inputs."""
    baseline = Config.BASELINES[model]
    adapter = importlib.import_module(
        f"rnagym.s3d.models.{baseline.environment}"
    ).ADAPTER
    return frozenset(
        target.parent.name if isinstance(target, Path) else target
        for target in adapter.targets(kind)
        if adapter.complete(target)
    )


@cache
def _valid_prediction(prediction: Path) -> bool:
    """Check one prediction shared by repeated experimental structures."""
    return valid_prediction(prediction)


def prediction_available(target: dict, model: str) -> bool:
    """Check for a completed prediction without loading its reference structure."""
    multimer = target["type"] == "multimer"
    kind = "multimers" if multimer else "monomers"
    name = target["PDB ID"] if multimer else target["sequence_id"]
    _, prediction = Config.get_bl_out_pdb(model, name, multimer)
    return name.lower() in _completed_targets(model, kind) and _valid_prediction(
        prediction
    )


def score_target(job: tuple[dict, str]) -> dict:
    """Score one model against one experimental target."""
    target, model = job
    analysis = Analysis(
        target["PDB ID"],
        target["Asym. Chain ID"],
        target["Auth. Chain ID"],
        target["sequence_id"],
    )
    baseline = Config.BASELINES[model]
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
    """Return every available target and model pair."""
    jobs = []
    for target in targets:
        jobs.extend(
            (target, model)
            for model, baseline in Config.BASELINES.items()
            if (target["type"] == "monomer" or baseline.mul_afa_file is not None)
            and prediction_available(target, model)
        )
    # Group by model so workers annotate distinct references instead of sharing locks
    return sorted(jobs, key=lambda job: (job[1], -job[0]["L"]))


def main() -> None:
    """Score available predictions and write the shared score table."""
    targets = pd.read_parquet(Config3D.TARGET_FILE).to_dict("records")
    jobs = scoring_jobs(targets)
    if not jobs:
        print("No available predictions to score")
        return
    missing_models = set(Config.BASELINES) - {model for _, model in jobs}
    if missing_models:
        print(f"No predictions found for: {', '.join(sorted(missing_models))}")
    if not Config3D.MC_ANNOTATE.is_file():
        raise FileNotFoundError(f"Missing MC-Annotate: {Config3D.MC_ANNOTATE}")
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(score_target, job) for job in jobs]
        progress = tqdm(
            as_completed(futures), total=len(futures), desc="Scoring predictions"
        )
        results = [future.result() for future in progress]

    scores = pd.DataFrame(results).sort_values(SCORE_KEY)
    if scores.duplicated(SCORE_KEY).any():
        raise RuntimeError("Duplicate 3D scores")
    temporary = Config3D.SCORE_FILE.with_suffix(".tmp.parquet")
    scores.to_parquet(temporary, index=False)
    temporary.replace(Config3D.SCORE_FILE)
    print(f"Wrote {len(scores)} scores to {Config3D.SCORE_FILE}")


if __name__ == "__main__":
    main()
