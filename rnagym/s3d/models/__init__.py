"""3D structure prediction model adapters and metadata."""

from dataclasses import dataclass
from pathlib import Path

from rnagym.config import Config3D


@dataclass(frozen=True)
class Baseline:
    """Describe one evaluated 3D structure predictor."""

    name: str
    environment: str
    prediction: str
    supports_multimers: bool
    training_cutoff: str


BASELINES = {
    baseline.name: baseline
    for baseline in (
        # https://doi.org/10.1038/s41586-024-07487-w
        Baseline("af3", "af3", "{name}/{name}_model.cif", True, "2021-09-30"),
        # https://doi.org/10.1038/s41467-025-56261-7
        Baseline(
            "nu",
            "nufold",
            "output/{name}/{name}_rank_1.pdb",
            False,
            "2022-02-28",
        ),
        # https://doi.org/10.1038/s41592-023-02086-5
        Baseline("rf2na", "rf2na", "models/model_00.pdb", True, "2020-04-30"),
        # https://doi.org/10.1038/s41592-024-02487-0
        Baseline(
            "rho",
            "rhofold",
            "output/unrelaxed_model.pdb",
            False,
            "2022-04-13",
        ),
        # https://doi.org/10.1038/s41467-023-42528-4
        Baseline("trRNA", "trrna", "model_1.pdb", False, "2022-01-01"),
    )
}


def homology_columns(model: str) -> tuple[str, str, str, str]:
    """Return the structural training-homology columns for one model."""
    prefix = f"{model.lower()}_tm_homolog"
    return prefix, f"{prefix}_date", f"{prefix}_rfam", f"{prefix}_score"


def prediction_path(model: str, name: str, multimer: bool) -> tuple[Path, Path]:
    """Return one model output directory and primary structure file."""
    kind = "multimers" if multimer else "monomers"
    name = name.lower()
    output_dir = (Config3D.PREDICTION_DIR / model / kind / name).resolve()
    prediction = output_dir / BASELINES[model].prediction.format(name=name)
    return output_dir, prediction
