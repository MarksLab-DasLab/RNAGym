import sys
from functools import cache
from itertools import product

import numpy as np
import torch

MODEL_SOURCE = ".pixi/model-sources/UFold"
MODEL_WEIGHTS = ".pixi/model-weights/ufold/ufold_train_alldata.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Upstream input construction:
# https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold/data_generator.py#L524-L552
# https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold/data_generator.py#L858-L864
MIN_INPUT_LENGTH = 80
INPUT_MULTIPLE = 16
INPUT_CHANNELS = 17
# Upstream post-processing settings:
# https://github.com/uci-cbcl/UFold/blob/75bd9acc83826059682dfca9d3659df66b132cd1/ufold_predict.py#L178-L185
sys.path.insert(0, MODEL_SOURCE)
from Network import U_Net  # noqa: E402
from ufold.postprocess import postprocess_new  # noqa: E402
from ufold.utils import creatmat, seq_encoding  # noqa: E402


@cache
def _model():
    model = U_Net(img_ch=INPUT_CHANNELS)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def _inputs(sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
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


def fold(sequence: str) -> np.ndarray:
    features, encoded = _inputs(sequence)
    with torch.no_grad():
        logits = _model()(features)
        contacts = postprocess_new(
            u=logits,
            x=encoded,
            lr_min=0.01,
            lr_max=0.1,
            num_itr=100,
            rho=1.6,
            with_l1=True,
            s=1.5,
        )
    return contacts[0, : len(sequence), : len(sequence)].cpu().numpy()
