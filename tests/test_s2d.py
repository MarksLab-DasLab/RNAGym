"""Test secondary structure scoring, decoders, and model adapters."""

import importlib
import os

import numpy as np
import polars as pl
import pytest
from rnagym.s2d.models.utils import (
    decode_pair_probabilities,
    pairs_to_dot_bracket,
    parse_pairs,
)

DECODERS = {"hungarian", "threshknot"}
MODEL = os.environ.get("PIXI_ENVIRONMENT_NAME")
MODEL_METHODS = {
    "contrafold": {"viterbi", "mea_gamma_1"},
    "eternafold": {"viterbi", "mea_gamma_1"},
    "mxfold2": {"mfe"},
    "ribonanzanet": set(),
    "rinalmo": {"greedy"},
    "rna-fm": {"postprocess_0.5"},
    "rnastructure": {"mfe", "mea_gamma_1", "probknot"},
    "ufold": {"threshold_0.5"},
    "vienna": {"mfe", "mea_gamma_1", "rnapkplex"},
}
SEQUENCE = "GGGGAAAACCCC"


def test_adapter() -> None:
    """Check one model adapter's output contract."""
    if MODEL not in MODEL_METHODS:
        pytest.skip("Run through a model environment")
    if MODEL == "rinalmo":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("RiNALMo requires a GPU")
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

    if MODEL == "vienna":
        structures = {
            structure["method"]: structure["dot_bracket"]
            for structure in adapter.predict("G")["structures"]
        }
        assert structures["rnapkplex"] == "."


def test_decoders() -> None:
    """Decode crossing helices and extended dot-bracket notation."""
    probabilities = np.zeros((12, 12))
    expected = {(0, 5), (1, 4), (2, 9), (3, 8)}
    for i, j in expected:
        probabilities[i, j] = probabilities[j, i] = 0.9
    for structure in decode_pair_probabilities(probabilities):
        assert parse_pairs(structure["dot_bracket"]) == expected

    expected = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}
    assert parse_pairs("([{<A)]}>a") == expected
    assert parse_pairs("([{<a)]}>A") == expected

    expected = {(0, 5), (1, 4), (2, 7), (3, 6)}
    assert parse_pairs(pairs_to_dot_bracket(8, expected)) == expected


def test_scoring() -> None:
    """Check modifier masks, resolution masks, and cluster macro-averaging."""
    pytest.importorskip("scipy")
    from rnagym.s2d.tasks.score import (
        score_mapping_batch,
        structure_f1,
        summarize,
    )

    scores = pl.DataFrame(
        {
            "model": ["model"] * 6,
            "dataset": ["2d"] * 3 + ["mapping"] * 3,
            "modality": ["pdb"] * 3 + ["DMS"] * 3,
            "method": ["method"] * 6,
            "metric": ["f1"] * 3 + ["spearman"] * 3,
            "cluster_rep": ["cluster"] * 6,
            "sequence_id": ["a", "a", "b"] * 2,
            "score": [0.2, 0.8, 0.4] * 2,
        }
    )
    summary = summarize(scores)
    structures = summary.filter(pl.col("dataset") == "2d").row(0, named=True)
    mapping = summary.filter(pl.col("dataset") == "mapping").row(0, named=True)
    assert structures["score"] == pytest.approx(0.6)
    assert structures["samples"] == 2
    assert mapping["score"] == pytest.approx(1.4 / 3)
    assert mapping["samples"] == 3

    batch = pl.DataFrame(
        {
            "sequence": ["ACGU"],
            "reactivity": [[0.1, 0.2, 100.0, -100.0]],
            "probabilities": [[0.9, 0.8, 0.0, 1.0]],
        }
    )
    assert score_mapping_batch(batch, "DMS", 4)[0] == pytest.approx(1.0)
    assert score_mapping_batch(batch, "CMCT", 4)[0] == pytest.approx(1.0)
    assert structure_f1("....", "....") == 1.0
    assert structure_f1("(())", "....") == 0.0
    assert structure_f1("(())", ".().", [False, True, True, False]) == 1.0
