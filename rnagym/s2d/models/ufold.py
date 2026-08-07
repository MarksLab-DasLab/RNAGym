"""UFold model adapter."""

import sys
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .utils import Prediction, decode_pair_probabilities, dot_bracket

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = PROJECT_ROOT / ".pixi" / "model-sources" / "UFold"
MODEL_WEIGHTS = (
    PROJECT_ROOT / ".pixi" / "model-weights" / "ufold" / "ufold_train_alldata.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-processing
MIN_INPUT_LENGTH = 80
INPUT_MULTIPLE = 16
INPUT_CHANNELS = 17

# Postprocessing
sys.path.insert(0, str(MODEL_SOURCE))
from Network import U_Net  # noqa: E402
from ufold.postprocess import postprocess_new  # noqa: E402
from ufold.utils import creatmat, seq_encoding  # noqa: E402


@cache
def _model():
    """Load the pretrained UFold model."""
    # https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold_predict.py#L310-L314
    model = U_Net(img_ch=INPUT_CHANNELS)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    # NOTE(MCA): Official prediction inexplicably keeps the network in training mode...
    # https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold_predict.py#L145-L147
    model.train().to(DEVICE)
    return model


def _inputs(sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode and pad a sequence for UFold."""
    # Upstream input construction:
    # https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold/data_generator.py#L524-L552
    # https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold/data_generator.py#L858-L864
    encoded = seq_encoding(sequence)
    length = max(
        MIN_INPUT_LENGTH,
        ((len(sequence) + INPUT_MULTIPLE - 1) // INPUT_MULTIPLE) * INPUT_MULTIPLE,
    )
    padded = np.zeros((length, 4))
    padded[: len(sequence)] = encoded
    features = np.zeros((INPUT_CHANNELS, length, length))
    for channel, (i, j) in enumerate(product(range(4), repeat=2)):
        features[channel, : len(sequence), : len(sequence)] = np.outer(
            encoded[:, i], encoded[:, j]
        )
    features[16, : len(sequence), : len(sequence)] = creatmat(sequence)
    return (
        torch.tensor(features[None], dtype=torch.float32, device=DEVICE),
        torch.tensor(padded[None], dtype=torch.float32, device=DEVICE),
    )


def _pair_probabilities(sequence: str) -> np.ndarray:
    """Return UFold's postprocessed pair probabilities."""
    features, encoded = _inputs(sequence)
    # https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold_predict.py#L177-L185
    with torch.no_grad():
        logits = _model()(features)
        pair_probabilities = postprocess_new(
            u=logits,
            x=encoded,
            lr_min=0.01,
            lr_max=0.1,
            num_itr=100,
            rho=1.6,
            with_l1=True,
            s=1.5,
        )
    return pair_probabilities[0, : len(sequence), : len(sequence)].cpu().numpy()


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and decoded structures."""
    pair_probabilities = _pair_probabilities(sequence)
    probabilities = np.clip(pair_probabilities.sum(axis=0), 0, 1)

    # Official inference decodes the postprocessed contacts at 0.5
    # https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold_predict.py#L182-L185
    # This differs VERY SLIGHTLY from UFold by resolving competing pairs by confidence
    candidates = np.argwhere(np.triu(pair_probabilities, k=1) > 0.5)
    candidates = candidates[np.argsort(pair_probabilities[tuple(candidates.T)])[::-1]]
    contacts = np.zeros_like(pair_probabilities, dtype=bool)
    used = set()
    for i, j in candidates:
        if i not in used and j not in used:
            contacts[i, j] = True
            used.update((i, j))
    structure = dot_bracket(contacts)
    structures = [{"method": "threshold_0.5", "dot_bracket": structure}]
    structures += decode_pair_probabilities(pair_probabilities)
    return {
        "probabilities": probabilities.tolist(),
        "structures": structures,
    }
