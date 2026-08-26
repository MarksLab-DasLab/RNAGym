"""RibonanzaNet model adapter."""

import sys

import numpy as np
import pandas as pd
import torch

from rnagym.config import Config2D

from .utils import Prediction, decode_pair_probabilities

MODEL_SOURCE = Config2D.MODEL_SOURCE_DIR / "rnet-inference" / "src"

sys.path.insert(0, str(MODEL_SOURCE))
# Official model definition and checkpoint loading
# https://github.com/DasLab/rnet-inference/blob/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/src/rnet_2d.py#L12-L31
from rnet_2d import RNA_Dataset  # noqa: E402
from rnet_2d import model as MODEL  # noqa: E402


def _pair_probabilities(sequence: str) -> np.ndarray:
    """Return pair probabilities."""
    # Official sequence encoding and inference
    # https://github.com/DasLab/rnet-inference/blob/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/src/rnet_2d.py#L75-L90
    encoded = RNA_Dataset(pd.DataFrame([{"sequence": sequence}]))[0]["sequence"]
    device = next(MODEL.parameters()).device
    encoded = encoded.unsqueeze(0).to(device)
    with torch.no_grad():
        pair_probabilities = MODEL(encoded).sigmoid()[0]
    return pair_probabilities.cpu().numpy()


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and decoded structures."""
    # Single-sequence inference is ~40 sequences/s on an L40S, so batching is unnecessary
    pair_probabilities = _pair_probabilities(sequence)

    # Official inference ignores pairs separated by fewer than four positions
    # https://github.com/DasLab/rnet-inference/blob/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/src/rnet_2d.py#L97-L110
    positions = np.arange(len(sequence))
    pair_probabilities[np.abs(positions[:, None] - positions[None, :]) < 4] = 0

    # Official Arnie decoding clips summed pair confidences before calculating unpaired probabilities
    # https://github.com/WaymentSteeleLab/arnie/blob/660de8139bd2198bbe115adadd5bc5f12183f9f4/src/arnie/pk_predictors.py#L111-L116
    probabilities = np.clip(pair_probabilities.sum(axis=0), 0, 1)

    return {
        "probabilities": probabilities.tolist(),
        "structures": decode_pair_probabilities(pair_probabilities),
    }
