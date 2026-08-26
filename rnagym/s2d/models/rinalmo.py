"""RiNALMo model adapter."""

import sys
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .utils import Prediction, decode_pair_probabilities, dot_bracket

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = PROJECT_ROOT / ".pixi" / "model-sources" / "RiNALMo"
MODEL_WEIGHTS = (
    PROJECT_ROOT / ".pixi" / "model-weights" / "rinalmo" / "rinalmo_giga_ss_bprna_ft.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_PAIR_CELLS = 2_000_000
MAX_BATCH_SIZE = 256

sys.path.insert(0, str(MODEL_SOURCE))
from rinalmo.config import model_config  # noqa: E402
from rinalmo.data.alphabet import Alphabet  # noqa: E402
from rinalmo.model.downstream import SecStructPredictionHead  # noqa: E402
from rinalmo.model.model import RiNALMo  # noqa: E402


# Official fine-tuned model architecture
# https://github.com/lbcb-sci/RiNALMo/blob/2c2c5c14a5ae609d8c560a5d9ca32e51e0288955/train_sec_struct_prediction.py#L26-L55
class _Model(nn.Module):
    """Match the official fine-tuned model and checkpoint names."""

    def __init__(self) -> None:
        super().__init__()
        self.lm = RiNALMo(model_config("giga"))
        self.pred_head = SecStructPredictionHead(
            self.lm.config.model.transformer.embed_dim, num_blocks=2
        )
        self.threshold = 0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return symmetric pair logits without the CLS and EOS tokens."""
        representation = self.lm(tokens)["representation"][:, 1:-1]
        return self.pred_head(representation)


def _allowed_pairs(sequence: str) -> np.ndarray:
    """Mask sharp loops and noncanonical pairs as in official inference."""
    bases = np.frombuffer(sequence.replace("U", "T").encode(), dtype=np.uint8)
    left, right = bases[:, None], bases[None, :]
    canonical = (
        ((left == ord("A")) & (right == ord("T")))
        | ((left == ord("T")) & (right == ord("A")))
        | ((left == ord("G")) & (right == ord("C")))
        | ((left == ord("C")) & (right == ord("G")))
        | ((left == ord("G")) & (right == ord("T")))
        | ((left == ord("T")) & (right == ord("G")))
    )
    positions = np.arange(len(sequence))
    return canonical & (np.abs(positions[:, None] - positions[None, :]) >= 4)


@cache
def _alphabet() -> Alphabet:
    """Return the model's official tokenizer."""
    return Alphabet(**_model().lm.config.alphabet)


def _decode(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Greedily resolve pairs above the checkpoint's learned threshold."""
    # Official thresholding and greedy conflict resolution
    # https://github.com/lbcb-sci/RiNALMo/blob/2c2c5c14a5ae609d8c560a5d9ca32e51e0288955/rinalmo/utils/sec_struct.py#L130-L167
    scores = np.where(probabilities > threshold, probabilities, 0)
    contacts = np.zeros_like(scores, dtype=bool)
    while np.any(scores):
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        scores[i, :] = scores[j, :] = 0
        scores[:, i] = scores[:, j] = 0
        contacts[i, j] = contacts[j, i] = True
    return contacts


@cache
def _model() -> _Model:
    """Load the official bpRNA-1m fine-tuned Giga checkpoint."""
    if DEVICE.type != "cuda":
        raise RuntimeError("RiNALMo requires a CUDA GPU")
    model = _Model()
    state = torch.load(MODEL_WEIGHTS, map_location="cpu")
    model.threshold = float(state.pop("threshold"))
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model


def _pair_probabilities(sequences: Sequence[str]) -> list[np.ndarray]:
    """Return pair probabilities after the official biological masks."""
    tokens = torch.tensor(
        _alphabet().batch_tokenize(sequences), dtype=torch.int64, device=DEVICE
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = _model()(tokens)
    batch = torch.sigmoid(logits).float().cpu().numpy()
    probabilities = []
    for sequence, matrix in zip(sequences, batch, strict=True):
        matrix = matrix[: len(sequence), : len(sequence)]
        matrix[~_allowed_pairs(sequence)] = 0
        probabilities.append(matrix)
    return probabilities


def _prediction(sequence: str, pair_probabilities: np.ndarray) -> Prediction:
    """Convert one pair matrix into benchmark outputs."""
    structure = dot_bracket(_decode(pair_probabilities, _model().threshold))
    structures = [{"method": "greedy", "dot_bracket": structure}]
    # fp16 scores contain exact ties that make Arnie assign one residue twice
    # Break ties below float32 precision before applying the shared decoders
    order = np.arange(pair_probabilities.size).reshape(pair_probabilities.shape)
    tie_break = np.minimum(order, order.T) * np.finfo(np.float64).eps
    structures += decode_pair_probabilities(pair_probabilities + tie_break)
    return {
        "probabilities": np.clip(pair_probabilities.sum(axis=0), 0, 1).tolist(),
        "structures": structures,
    }


def batch_size(sequence_length: int) -> int:
    """Choose a batch size with bounded quadratic head memory."""
    return min(MAX_BATCH_SIZE, max(1, MAX_BATCH_PAIR_CELLS // sequence_length**2))


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and decoded structures."""
    return predict_batch([sequence])[0]


def predict_batch(sequences: Sequence[str]) -> list[Prediction]:
    """Predict a batch of equal-length sequences."""
    if not sequences:
        return []
    # Padding changes the structure head's instance normalization
    if len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError("RiNALMo batches must contain equal-length sequences")
    if len(sequences[0]) == 1:
        # A one-base RNA cannot contain a base pair
        probabilities = [np.zeros((1, 1), dtype=np.float32) for _ in sequences]
    else:
        probabilities = _pair_probabilities(sequences)
    return [
        _prediction(sequence, matrix)
        for sequence, matrix in zip(sequences, probabilities, strict=True)
    ]
