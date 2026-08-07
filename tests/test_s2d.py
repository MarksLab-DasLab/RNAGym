"""Test secondary structure decoders and model adapters."""

import importlib
import os

import numpy as np
import pytest
from rnagym.s2d.models.utils import decode_pair_probabilities
from rnagym.s2d.tasks.score import parse_pairs

SEQUENCE = "GGGGAAAACCCC"
MODEL = os.environ.get("PIXI_ENVIRONMENT_NAME")
DECODERS = {"hungarian", "threshknot"}
MODEL_METHODS = {
    "vienna": {"mfe", "mea_gamma_1", "rnapkplex"},
    "contrafold": {"viterbi", "mea_gamma_1"},
    "eternafold": {"viterbi", "mea_gamma_1"},
    "rnastructure": {"mfe", "mea_gamma_1", "probknot"},
    "ribonanzanet": set(),
    "ufold": {"threshold_0.5"},
    "rna-fm": {"postprocess_0.5"},
    "mxfold2": {"mfe"},
}


def test_shared_decoders() -> None:
    """Recover crossing helices from a known pair-probability matrix."""
    probabilities = np.zeros((12, 12))
    expected = {(0, 5), (1, 4), (2, 9), (3, 8)}
    for i, j in expected:
        probabilities[i, j] = probabilities[j, i] = 0.9
    for structure in decode_pair_probabilities(probabilities):
        assert parse_pairs(structure["dot_bracket"]) == expected


def test_adapter() -> None:
    """Check one model adapter's output contract."""
    if MODEL not in MODEL_METHODS:
        pytest.skip("Run through a model environment")
    adapter = importlib.import_module(f"rnagym.s2d.models.{MODEL.replace('-', '_')}")
    prediction = adapter.predict(SEQUENCE)
    probabilities = np.asarray(prediction["probabilities"])
    assert probabilities.shape == (len(SEQUENCE),)
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    methods = {structure["method"] for structure in prediction["structures"]}
    assert methods == DECODERS | MODEL_METHODS[MODEL]
    for structure in prediction["structures"]:
        assert len(structure["dot_bracket"]) == len(SEQUENCE)
        parse_pairs(structure["dot_bracket"])
