import os
from functools import cache

import numpy as np
import torch

MODEL_WEIGHTS = ".pixi/model-weights/rna-fm/RNA-FM-ResNet_PDB-All.pth"
TORCH_HOME = ".pixi/model-weights/rna-fm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@cache
def _model():
    from fm.downstream.baseline import Baseline

    os.environ["TORCH_HOME"] = TORCH_HOME
    # NOTE(MCA): The packaged loader has a broken URL and cannot load its
    # checkpoint on CPU.
    # https://github.com/ml4bio/RNA-FM/blob/348951516e0963d22bbb33b3c9fc18c89081d38e/fm/downstream/__init__.py#L22-L42
    model = Baseline(
        backbone_name="rna-fm",
        pairwise_predictor_name="pc-resnet_1_sym_first:r-ss",
        backbone_frozen=1,
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def fold(sequence: str) -> np.ndarray:
    model = _model()
    converter = model.backbone_alphabet.get_batch_converter()
    _, _, tokens = converter([("sequence", sequence)])
    with torch.no_grad():
        logits = model({"token": tokens.to(DEVICE)})["r-ss"]
    # Official inference converts r-ss logits with sigmoid:
    # https://github.com/ml4bio/RNA-FM/blob/348951516e0963d22bbb33b3c9fc18c89081d38e/redevelop/engine/predictor.py#L140-L153
    return torch.sigmoid(logits[0]).cpu().numpy()
