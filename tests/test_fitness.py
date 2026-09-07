"""Reproduce fitness evaluation and model scoring on real assay variants."""

import os
import shlex
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from rnagym.config import ConfigFitness
from rnagym.fitness.tasks.model_registry import SCORE_COLS, resolve_source

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fitness"
MODEL = os.environ.get("PIXI_ENVIRONMENT_NAME")


@pytest.fixture
def fitness_data(tmp_path, monkeypatch):
    """Write the fixture tables in the format used by the prediction commands."""
    for name, path in {
        "DATA_DIR": tmp_path,
        "ASSAY_DIR": tmp_path / "assays",
        "COMBINED_DIR": tmp_path / "merged",
        "LEADERBOARD_DIR": tmp_path / "leaderboard",
        "MSA_DIR": tmp_path / "msa",
        "PREDICTION_DIR": tmp_path / "predictions",
        "REFERENCE_FILE": FIXTURE_DIR / "reference.csv",
    }.items():
        monkeypatch.setattr(ConfigFitness, name, path)
    ConfigFitness.ASSAY_DIR.mkdir()
    for (name,), assay in pl.read_parquet(FIXTURE_DIR / "assays.parquet").group_by(
        "DMS_ID"
    ):
        assay.drop("DMS_ID").write_csv(ConfigFitness.ASSAY_DIR / f"{name}.csv")
    for (model, name), predictions in pl.read_parquet(
        FIXTURE_DIR / "predictions.parquet"
    ).group_by("model", "DMS_ID"):
        folder, column = resolve_source(SCORE_COLS, model)
        destination = ConfigFitness.PREDICTION_DIR / folder / f"{name}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        predictions.select("mutant", pl.col("score").alias(column)).write_csv(
            destination
        )
    return tmp_path


def test_evmutation_scoring(fitness_data, monkeypatch) -> None:
    """Compare real MSA predictions with the native model's sequence energies."""
    if MODEL != "evmutation":
        pytest.skip("Run through the EVmutation environment")
    from rnagym.fitness.baselines.EVmutation import compute_fitness as scorer

    name = "Domingo_2018_tRNA"
    alignment = ConfigFitness.MSA_DIR / "by_assay" / f"{name}.a3m"
    alignment.parent.mkdir(parents=True)
    alignment.symlink_to((FIXTURE_DIR / alignment.name).resolve())
    alignment.with_suffix(".fa").write_text(
        "\n".join(alignment.read_text().splitlines()[:2]) + "\n"
    )
    models = []
    infer = scorer.infer_model

    def capture_model(config):
        model = infer(config)
        models.append(model)
        return model

    monkeypatch.setattr(scorer, "infer_model", capture_model)
    output = fitness_data / "scored"
    monkeypatch.setattr(
        sys, "argv", shlex.split(f"evmutation --output {output} --cpu 2")
    )
    scorer.main()
    model = models[0]
    assay = pl.read_csv(ConfigFitness.ASSAY_DIR / f"{name}.csv")
    result = pl.read_csv(output / f"{name}.csv")
    assert_frame_equal(result.select(assay.columns), assay)
    reference = pl.read_csv(ConfigFitness.REFERENCE_FILE)
    wt = reference.filter(pl.col("DMS_ID") == name)["RAW_CONSTRUCT_SEQ"][0]
    positions = model.index_list - 1
    covered = np.array(
        [
            all(i in positions for i, (a, b) in enumerate(zip(wt, sequence)) if a != b)
            for sequence in assay["sequence"]
        ]
    )
    np.testing.assert_array_equal(result[scorer.SCORE_COLUMN].is_not_null(), covered)
    sequences = [wt, *assay.filter(pl.Series(covered))["sequence"]]
    energies = model.hamiltonians(
        ["".join(s[i] for i in positions).replace("U", "T") for s in sequences]
    )[:, 0]
    np.testing.assert_allclose(
        result.filter(pl.Series(covered))[scorer.SCORE_COLUMN],
        energies[1:] - energies[0],
        atol=1e-6,
        rtol=1e-6,
    )


def test_fitness_workflow(fitness_data) -> None:
    """Reproduce both leaderboards from released predictions for every model."""
    if MODEL not in {None, "default"}:
        pytest.skip("Run the shared workflow in the default environment")
    from rnagym.fitness.tasks import leaderboard, merge_scoring_files

    merge_scoring_files.main([])
    leaderboard.main([])
    for actual, expected in (
        ("leaderboard_signed_3ncRNA.csv", "leaderboard.csv"),
        ("leaderboard_evmutation.csv", "comparison.csv"),
        ("evmutation_coverage.csv", "coverage.csv"),
    ):
        assert_frame_equal(
            pl.read_csv(ConfigFitness.LEADERBOARD_DIR / actual),
            pl.read_csv(FIXTURE_DIR / expected),
            check_exact=False,
            rel_tol=0,
            abs_tol=1e-12,
        )
    assert (ConfigFitness.LEADERBOARD_DIR / "README.md").read_text() == (
        FIXTURE_DIR / "leaderboard.md"
    ).read_text()
    for source in ConfigFitness.ASSAY_DIR.glob("*.csv"):
        assay = pl.read_csv(source).drop_nulls("mutant")
        merged = pl.read_csv(ConfigFitness.COMBINED_DIR / source.name)
        assert_frame_equal(merged.select(assay.columns), assay)


def test_genslm_scoring(fitness_data) -> None:
    """Compare checkpoint predictions and padded batches with native causal losses."""
    if MODEL != "genslm":
        pytest.skip("Run through the GenSLM environment")
    import genslm
    import torch

    from rnagym.fitness.baselines.GenSLM import compute_fitness as scorer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = genslm.GenSLM(
        "genslm_2.5B_patric",
        model_cache_dir=str(ConfigFitness.CHECKPOINT_DIR / "genslm/2.5B"),
    )
    output = fitness_data / "scored"
    scorer.main(
        scorer.parse_args(shlex.split(f"--output {output}")),
        lambda *args, **kwargs: model,
    )
    sequences, expected, observed = [], [], []
    for source in sorted(ConfigFitness.ASSAY_DIR.glob("*.csv")):
        assay = pl.read_csv(source).drop_nulls("mutant")
        result = pl.read_csv(output / source.name)
        assert_frame_equal(result.select(assay.columns), assay)
        sequence = assay["sequence"][0].replace("U", "T")
        encoded = genslm.SequenceDataset(
            [sequence], (len(sequence) + 2) // 3, model.tokenizer, verbose=False
        )[0]
        with torch.inference_mode():
            loss = model(
                encoded["input_ids"][None].to(device),
                encoded["attention_mask"].to(device),
            ).loss.item()
        sequences.append(sequence)
        expected.append(-loss)
        observed.append(result["logit_scores"][0])
    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(
        scorer.sequence_log_likelihoods(model, sequences, device),
        expected,
        atol=1e-5,
        rtol=1e-6,
    )
