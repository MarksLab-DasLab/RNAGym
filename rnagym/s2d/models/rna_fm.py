"""RNA-FM model adapter."""

import os
from functools import cache

import numpy as np
import torch

from rnagym.config import Config2D

from .utils import Prediction, decode_pair_probabilities, dot_bracket

MODEL_WEIGHTS = Config2D.CHECKPOINT_DIR / "rna-fm" / "RNA-FM-ResNet_PDB-All.pth"
TORCH_HOME = Config2D.CHECKPOINT_DIR / "rna-fm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@cache
def _model():
    """Load the pretrained RNA-FM model."""
    from fm.downstream.baseline import Baseline

    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    # The packaged loader has a broken URL and cannot load its checkpoint on CPU
    # https://github.com/ml4bio/RNA-FM/blob/348951516e0963d22bbb33b3c9fc18c89081d38e/fm/downstream/__init__.py#L22-L42
    model = Baseline(
        backbone_name="rna-fm",
        pairwise_predictor_name="pc-resnet_1_sym_first:r-ss",
        backbone_frozen=1,
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def _pair_probabilities(sequence: str) -> np.ndarray:
    """Return RNA-FM pair probabilities."""
    model = _model()
    converter = model.backbone_alphabet.get_batch_converter()
    _, _, tokens = converter([("sequence", sequence)])
    with torch.no_grad():
        logits = model({"token": tokens.to(DEVICE)})["r-ss"]
    # Official inference converts r-ss logits with sigmoid:
    # https://github.com/ml4bio/RNA-FM/blob/348951516e0963d22bbb33b3c9fc18c89081d38e/redevelop/engine/predictor.py#L140-L153
    return torch.sigmoid(logits[0]).cpu().numpy()


def _decode(pair_probabilities: np.ndarray) -> np.ndarray:
    """Apply RNA-FM's official greedy postprocessor."""
    # Official decoding thresholds at 0.5, removes self and adjacent pairs,
    # and greedily keeps the highest-scoring nonconflicting pairs
    # https://github.com/ml4bio/RNA-FM/blob/348951516e0963d22bbb33b3c9fc18c89081d38e/redevelop/engine/predictor.py#L280-L327
    pair_probabilities = pair_probabilities.copy()
    np.fill_diagonal(pair_probabilities, 0)
    candidates = np.argwhere(pair_probabilities > 0.5)
    order = np.argsort(-pair_probabilities[tuple(candidates.T)])
    contacts = np.zeros_like(pair_probabilities, dtype=bool)
    # Resolve directed assignments into valid undirected pairs by confidence
    used = set()
    for i, j in candidates[order]:
        if abs(i - j) <= 1 or i in used or j in used:
            continue
        contacts[i, j] = True
        used.update((i, j))
    return contacts


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and decoded structures."""
    # RNA-FM reserves two of its 1,024 positions for BOS/EOS
    if len(sequence) + 2 > _model().backbone.args.max_positions:
        return {
            "probabilities": [float("nan")] * len(sequence),
            "structures": [
                {"method": method, "dot_bracket": None}
                for method in ("postprocess_0.5", "hungarian", "threshknot")
            ],
        }
    pair_probabilities = _pair_probabilities(sequence)
    structure = dot_bracket(_decode(pair_probabilities))
    positions = np.arange(len(sequence))
    pair_probabilities[np.abs(positions[:, None] - positions[None, :]) <= 1] = 0
    probabilities = np.clip(pair_probabilities.sum(axis=0), 0, 1)
    structures = [{"method": "postprocess_0.5", "dot_bracket": structure}]
    structures += decode_pair_probabilities(pair_probabilities)
    return {
        "probabilities": probabilities.tolist(),
        "structures": structures,
    }
