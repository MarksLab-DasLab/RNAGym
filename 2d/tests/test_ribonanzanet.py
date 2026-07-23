import subprocess
import sys
from pathlib import Path

import numpy as np

from models.ribonanzanet import predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RIBONANZANET_SCRIPT = (
    PROJECT_ROOT / ".pixi" / "model-sources" / "rnet-inference" / "src" / "rnet_2d.py"
)
SEQUENCE = "GGGGAAAACCCC"

result = subprocess.run(
    [sys.executable, RIBONANZANET_SCRIPT, SEQUENCE, "--output-confidence"],
    check=True,
    capture_output=True,
    text=True,
)
# The official CLI prints its pair probability matrix as one flattened line
# https://github.com/DasLab/rnet-inference/blob/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/src/rnet_2d.py#L137-L138
official = np.fromstring(
    next(
        line.removeprefix("pair_confidence:")
        for line in result.stdout.splitlines()
        if line.startswith("pair_confidence:")
    ),
    sep=",",
).reshape(len(SEQUENCE), len(SEQUENCE))
official_structure = next(
    line.removeprefix("structure:")
    for line in result.stdout.splitlines()
    if line.startswith("structure:")
)

prediction = predict(SEQUENCE)
probabilities = prediction["probabilities"]
assert len(probabilities) == len(SEQUENCE)
assert prediction["structures"] == [
    {"method": "hungarian", "dot_bracket": official_structure}
]

# Apply the official near diagonal mask before the RNAGym row sum
# https://github.com/DasLab/rnet-inference/blob/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/src/rnet_2d.py#L97-L110
positions = np.arange(len(SEQUENCE))
official[np.abs(positions[:, None] - positions[None, :]) < 4] = 0
np.testing.assert_allclose(probabilities, official.sum(axis=1))
