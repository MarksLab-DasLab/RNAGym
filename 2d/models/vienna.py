"""ViennaRNA model adapter."""

import numpy as np
import RNA

from models.utils import Prediction, warn_out_of_range


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and MFE and MEA structures."""
    fold = RNA.fold_compound(sequence.replace("T", "U"))
    mfe_structure, mfe = fold.mfe()
    fold.exp_params_rescale(mfe)
    fold.pf()
    pair_probabilities = np.asarray(fold.bpp())
    mea_structure, _ = fold.MEA()

    # The matrix is one-indexed and stores each pair only in its upper triangle
    probabilities = pair_probabilities.sum(axis=0) + pair_probabilities.sum(axis=1)
    probabilities = probabilities[1:]
    warn_out_of_range(probabilities)
    return {
        "probabilities": probabilities.tolist(),
        "structures": [
            {"method": "mfe", "dot_bracket": mfe_structure},
            {"method": "mea_gamma_1", "dot_bracket": mea_structure},
        ],
    }
