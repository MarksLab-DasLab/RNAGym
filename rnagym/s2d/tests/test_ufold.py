"""Test the UFold model adapter."""

from ..models.ufold import predict

SEQUENCE = "GGGGAAAACCCC"
EXPECTED = "((((....))))"

prediction = predict(SEQUENCE)
probabilities = prediction["probabilities"]
assert len(probabilities) == len(SEQUENCE)
assert all(0 <= probability <= 1 for probability in probabilities)
assert prediction["structures"] == [
    {"method": "threshold_0.5", "dot_bracket": EXPECTED}
]
print(EXPECTED)
