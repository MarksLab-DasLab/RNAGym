"""Arnie model adapter."""

import numpy as np
from arnie.bpps import bpps


def predict(sequence: str, model: str) -> list[float]:
    """Return the probability that each nucleotide is paired."""
    pair_probabilities = bpps(sequence.replace("T", "U"), package=model)
    return np.clip(pair_probabilities.sum(axis=1), 0, 1).tolist()
