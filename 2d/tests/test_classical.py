import importlib
import sys

SEQUENCES = [
    "GGGGAAAACCCC",
    "GGGAACGACUCGAGUAGAGUCGAAAAACCUUGAUGGUGUGAAGCGUACUACUAUAGAUCCUGAUCUUAG"
    "UAAAUACGUGAAGCGUGGUGACUACUACUUUACAGCUGCACCCCUAGAUGUGGUGGCGUUAUCUAAUU"
    "CGUUCGCGAAUUAGAUAACAAAAGAAACAACAACAACAAC"
]

predict = importlib.import_module(f"models.{sys.argv[1]}").predict
for sequence in SEQUENCES:
    probabilities = predict(sequence)
    assert len(probabilities) == len(sequence)
    assert all(0 <= probability <= 1 for probability in probabilities)
