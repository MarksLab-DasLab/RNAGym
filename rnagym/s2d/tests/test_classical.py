"""Test the classical model adapters."""

import importlib
import sys

SEQUENCES = [
    "GGGGAAAACCCC",
    "GGGAACGACUCGAGUAGAGUCGAAAAACCUUGAUGGUGUGAAGCGUACUACUAUAGAUCCUGAUCUUAG"
    "UAAAUACGUGAAGCGUGGUGACUACUACUUUACAGCUGCACCCCUAGAUGUGGUGGCGUUAUCUAAUU"
    "CGUUCGCGAAUUAGAUAACAAAAGAAACAACAACAACAAC",
]

model = sys.argv[1]
predict = importlib.import_module(f"rnagym.s2d.models.{model}").predict
if model in {"contrafold", "eternafold"}:
    expected_methods = ["viterbi", "mea_gamma_1"]
elif model in {"vienna", "rnastructure"}:
    expected_methods = ["mfe", "mea_gamma_1"]
else:
    raise ValueError(f"Unknown classical model: {model}")

for sequence in SEQUENCES:
    prediction = predict(sequence)
    probabilities = prediction["probabilities"]
    assert len(probabilities) == len(sequence)
    assert all(0 <= probability <= 1 for probability in probabilities)
    assert [structure["method"] for structure in prediction["structures"]] == (
        expected_methods
    )
    for structure in prediction["structures"]:
        assert len(structure["dot_bracket"]) == len(sequence)
        assert set(structure["dot_bracket"]) <= {".", "(", ")"}
