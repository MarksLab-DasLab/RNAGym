"""Test the RNA-FM model adapter."""

import numpy as np

from ..models.rna_fm import predict

SEQUENCE = "GGGGAAAACCCC"
EXPECTED = "((((....))))"

prediction = predict(SEQUENCE)
probabilities = prediction["probabilities"]
assert len(probabilities) == len(SEQUENCE)
assert np.isfinite(probabilities).all()
assert all(0 <= probability <= 1 for probability in probabilities)
assert prediction["structures"] == [
    {"method": "postprocess_0.5", "dot_bracket": EXPECTED}
]
print(EXPECTED)
