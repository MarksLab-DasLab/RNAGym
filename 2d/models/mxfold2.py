"""MXFold2 model adapter."""

from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import mxfold2
import numpy as np
import torch
from mxfold2.predict import Predict

from models.utils import Prediction, warn_out_of_range

MODEL_DIR = Path(mxfold2.__file__).parent / "models"
CONFIG = MODEL_DIR / "TrainSetAB.conf"


@lru_cache(maxsize=1)
def _model():
    """Load MXFold2's default TrainSetAB model."""
    parser = ArgumentParser(fromfile_prefix_chars="@")
    Predict.add_args(parser.add_subparsers())
    args = parser.parse_args(["predict", "input.fa", f"@{CONFIG}"])

    predictor = Predict()
    model, _ = predictor.build_model(args)
    weights = torch.load(MODEL_DIR / args.param, map_location="cpu")
    model.load_state_dict(weights)
    model.eval()
    return model


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and MXFold2's official MFE structure."""
    # MXFold2 0.1.2 computes BPPs with its partition function
    # https://github.com/keio-bioinformatics/mxfold2/blob/51b213676708bebd664f0c40873a46e09353e1ee/mxfold2/fold/mix.py#L15-L60
    with torch.no_grad():
        _, structures, _, _, bpps = _model()([sequence], return_partfunc=True)

    bpp = np.triu(bpps[0])
    probabilities = (bpp + bpp.T).sum(axis=0)[1:]
    warn_out_of_range(probabilities)
    return {
        "probabilities": probabilities.tolist(),
        "structures": [{"method": "mfe", "dot_bracket": structures[0]}],
    }
