"""RibonanzaNet model adapter."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.utils import warn_out_of_range

MODEL_SOURCE = (
    Path(__file__).resolve().parents[1]
    / ".pixi"
    / "model-sources"
    / "rnet-inference"
    / "src"
)

sys.path.insert(0, str(MODEL_SOURCE))
# Official model definition and checkpoint loading
# https://github.com/DasLab/rnet-inference/blob/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/src/rnet_2d.py#L12-L31
from rnet_2d import RNA_Dataset, model as MODEL  # noqa: E402, I001


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


def predict(sequence: str) -> list[float]:
    """Return the probability that each nucleotide is paired."""
    # Single-sequence inference is ~40 sequences/s on an L40S, so batching is unnecessary
    pair_probabilities = _pair_probabilities(sequence)

    # Official inference ignores pairs separated by fewer than four positions
    # https://github.com/DasLab/rnet-inference/blob/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/src/rnet_2d.py#L97-L110
    positions = np.arange(len(sequence))
    pair_probabilities[np.abs(positions[:, None] - positions[None, :]) < 4] = 0

    # RNAGym chemical mapping scoring sums valid pair probabilities per nucleotide
    probabilities = pair_probabilities.sum(axis=1)
    warn_out_of_range(probabilities)
    return probabilities.tolist()
