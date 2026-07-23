"""EternaFold model adapter."""

from pathlib import Path

from models.contrafold import _predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ETERNAFOLD = PROJECT_ROOT / ".pixi" / "model-sources" / "EternaFold"


def predict(sequence: str) -> list[float]:
    """Return the probability that each nucleotide is paired."""
    return _predict(
        sequence,
        ETERNAFOLD / "src" / "contrafold",
        ETERNAFOLD / "parameters" / "EternaFoldParams.v1",
    )
