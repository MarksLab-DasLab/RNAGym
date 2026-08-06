"""EternaFold model adapter."""

from pathlib import Path

from .contrafold import _predict
from .utils import Prediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ETERNAFOLD = PROJECT_ROOT / ".pixi" / "model-sources" / "EternaFold"


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and Viterbi and MEA structures."""
    return _predict(
        sequence,
        ETERNAFOLD / "src" / "contrafold",
        ETERNAFOLD / "parameters" / "EternaFoldParams.v1",
    )
