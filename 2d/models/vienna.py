"""ViennaRNA model adapter."""

import numpy as np
import RNA

from models.utils import warn_out_of_range


def predict(sequence: str) -> list[float]:
    """Return the probability that each nucleotide is paired."""
    fold = RNA.fold_compound(sequence.replace("T", "U"))
    fold.pf()
    pair_probabilities = np.asarray(fold.bpp())

    # The matrix is one-indexed and stores each pair only in its upper triangle
    probabilities = pair_probabilities.sum(axis=0) + pair_probabilities.sum(axis=1)
    probabilities = probabilities[1:]
    warn_out_of_range(probabilities)
    return probabilities.tolist()
