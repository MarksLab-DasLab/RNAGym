"""ViennaRNA model adapter."""

import subprocess
import tempfile

import numpy as np
import RNA

from .utils import Prediction, decode_pair_probabilities, run, warn_out_of_range


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and decoded structures."""
    fold = RNA.fold_compound(sequence.replace("T", "U"))
    mfe_structure, mfe = fold.mfe()
    fold.exp_params_rescale(mfe)
    fold.pf()
    pair_probabilities = np.asarray(fold.bpp())
    mea_structure, _ = fold.MEA()

    with tempfile.TemporaryDirectory(prefix="rnagym-rnapkplex-") as tmpdir:
        try:
            result = run(
                "RNAPKplex",
                cwd=tmpdir,
                input=f">sequence\n{sequence}\n",
                text=True,
                stdout=subprocess.PIPE,
            )
            lines = result.stdout.splitlines()
            # RNAPKplex returns only the input FASTA when it predicts no pairs
            rnapkplex = lines[-1].split()[0] if len(lines) > 2 else "." * len(sequence)
        # RNAPKplex can overflow its partition function on long sequences
        except RuntimeError:
            rnapkplex = None

    # The matrix is one-indexed and stores each pair only in its upper triangle
    probabilities = pair_probabilities.sum(axis=0) + pair_probabilities.sum(axis=1)
    probabilities = probabilities[1:]
    warn_out_of_range(probabilities)
    probabilities = np.clip(probabilities, 0, 1)
    bpp = np.triu(pair_probabilities)[1:, 1:]
    structures = [
        {"method": "mfe", "dot_bracket": mfe_structure},
        {"method": "mea_gamma_1", "dot_bracket": mea_structure},
        {"method": "rnapkplex", "dot_bracket": rnapkplex},
    ]
    structures += decode_pair_probabilities(bpp + bpp.T)
    return {
        "probabilities": probabilities.tolist(),
        "structures": structures,
    }
