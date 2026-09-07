"""EternaFold model adapter."""

from rnagym.config import Config2D

from .contrafold import _predict
from .utils import Prediction

ETERNAFOLD = Config2D.MODEL_SOURCE_DIR / "EternaFold"


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and Viterbi and MEA structures."""
    return _predict(
        sequence,
        ETERNAFOLD / "src" / "contrafold",
        Config2D.CHECKPOINT_DIR / "eternafold" / "EternaFoldParams.v1",
    )
