"""Reproduce the 2D benchmark and model adapters on real RNA sequences."""

import importlib
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from rnagym.config import Config2D
from rnagym.s2d.models.utils import parse_pairs

DECODERS = {"hungarian", "threshknot"}
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "s2d"
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


def test_2d_workflow(tmp_path, monkeypatch) -> None:
    """Reproduce scoring and leaderboard output from released predictions."""
    if MODEL not in {None, "default"}:
        pytest.skip("Run the shared workflow in the default environment")
    from rnagym.s2d.tasks import score

    monkeypatch.setattr(Config2D, "MAPPING_FILE", FIXTURE_DIR / "mapping.parquet")
    monkeypatch.setattr(Config2D, "PREDICTION_DIR", FIXTURE_DIR / "predictions")
    monkeypatch.setattr(Config2D, "SEQUENCE_FILE", FIXTURE_DIR / "sequences.parquet")
    monkeypatch.setattr(Config2D, "STRUCTURE_FILE", FIXTURE_DIR / "structures.parquet")
    monkeypatch.setattr(Config2D, "LEADERBOARD_DIR", tmp_path)
    monkeypatch.setattr(Config2D, "LEADERBOARD_FILE", tmp_path / "leaderboard.csv")
    monkeypatch.setattr(Config2D, "LEADERBOARD_README", tmp_path / "README.md")
    Config2D.LEADERBOARD_README.write_text(
        "# Test leaderboard\n\n"
        "<!-- BEGIN GENERATED TABLE -->\n"
        "<!-- END GENERATED TABLE -->\n"
    )

    score.main()

    assert_frame_equal(
        pl.read_csv(Config2D.LEADERBOARD_FILE),
        pl.read_csv(FIXTURE_DIR / "leaderboard.csv"),
        check_exact=True,
    )
    assert (
        Config2D.LEADERBOARD_README.read_text()
        == (FIXTURE_DIR / "leaderboard.md").read_text()
    )


def test_model_adapter() -> None:
    """Check one model environment's prediction contract on a real RNA."""
    if MODEL not in MODEL_METHODS:
        pytest.skip("Run through a model environment")
    if MODEL == "rinalmo":
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            pytest.skip("RiNALMo requires a GPU")

    sequence = (
        pl.read_parquet(FIXTURE_DIR / "sequences.parquet", columns="sequence")
        .sort(pl.col("sequence").str.len_chars())
        .item(0, 0)
    )
    adapter = importlib.import_module(f"rnagym.s2d.models.{MODEL.replace('-', '_')}")
    prediction = adapter.predict(sequence)
    probabilities = np.asarray(prediction["probabilities"])
    assert probabilities.shape == (len(sequence),)
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    assert {structure["method"] for structure in prediction["structures"]} == (
        DECODERS | MODEL_METHODS[MODEL]
    )
    for structure in prediction["structures"]:
        assert len(structure["dot_bracket"]) == len(sequence)
        parse_pairs(structure["dot_bracket"])
