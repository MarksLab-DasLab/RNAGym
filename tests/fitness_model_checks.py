"""Check native fitness scoring in the documented per-model Pixi environments."""

import shlex
import sys
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from rnagym.config import ConfigFitness

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fitness"


def test_evmutation_scoring(tmp_path, monkeypatch):
    """Fit a real RNA MSA and verify coverage, coordinates and Potts energies."""
    from rnagym.fitness.baselines.EVmutation import compute_fitness as scorer
    from rnagym.fitness.tasks.merge_scoring_files import merge_predictions

    models = []
    fit = scorer.infer_model

    def capture_model(config):
        model = fit(config)
        models.append(model)
        return model

    monkeypatch.setattr(scorer, "infer_model", capture_model)
    monkeypatch.setattr(ConfigFitness, "ASSAY_DIR", FIXTURE_DIR / "assays")
    monkeypatch.setattr(ConfigFitness, "REFERENCE_FILE", FIXTURE_DIR / "reference.csv")
    monkeypatch.setattr(ConfigFitness, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ConfigFitness, "MSA_DIR", tmp_path / "msa")
    (tmp_path / "msa/by_assay").mkdir(parents=True)
    for path in (FIXTURE_DIR / "msa").iterdir():
        (tmp_path / "msa/by_assay" / path.name).symlink_to(path.resolve())
    monkeypatch.setattr(
        sys,
        "argv",
        shlex.split(f"evmutation --output {tmp_path / 'scores'} --cpu 2"),
    )
    scorer.main()
    assert len(models) == 1
    model = models[0]
    np.testing.assert_array_equal(model.index_list, np.arange(4, 69))
    assert model.N_valid == 991
    np.testing.assert_allclose(model.lambda_J, 0.01 * (model.L - 1) * 3, rtol=1e-7)

    name = "Domingo_2018_tRNA"
    assay = pl.read_csv(FIXTURE_DIR / "assays" / f"{name}.csv")
    result = pl.read_csv(tmp_path / "scores" / f"{name}.csv")
    assert_frame_equal(result.select(assay.columns), assay)
    reference = pl.read_csv(FIXTURE_DIR / "reference.csv")
    wt = reference.filter(pl.col("DMS_ID") == name)["RAW_CONSTRUCT_SEQ"][0]
    sequences = assay["sequence"].to_list()
    positions = model.index_list - 1
    covered = np.array(
        [
            all(i in positions for i, (a, b) in enumerate(zip(wt, sequence)) if a != b)
            for sequence in sequences
        ]
    )
    assert covered.sum() == 69
    np.testing.assert_array_equal(result[scorer.SCORE_COLUMN].is_not_null(), covered)
    projected = [
        "".join(sequence[i] for i in positions).replace("U", "T")
        for sequence in [wt] + [s for s, keep in zip(sequences, covered) if keep]
    ]
    native = model.hamiltonians(projected)[:, 0]
    tokens = model.convert_sequences(projected)
    energy = np.zeros(len(tokens))
    for i in range(model.L):
        energy += model.h_i[i, tokens[:, i]]
        for j in range(i + 1, model.L):
            energy += model.J_ij[i, j, tokens[:, i], tokens[:, j]]
    np.testing.assert_allclose(native, energy, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        result.filter(pl.Series(covered))[scorer.SCORE_COLUMN],
        energy[1:] - energy[0],
        atol=1e-6,
        rtol=1e-6,
    )
    merged = merge_predictions(assay, result, scorer.SCORE_COLUMN, "EVmutation")
    assert merged.height == assay.height
    assert merged["EVmutation_score"].null_count() == (~covered).sum()


def test_genslm_scoring(tmp_path, monkeypatch):
    """Compare real checkpoint predictions with native unpadded causal losses."""
    import genslm
    import torch
    from genslm import SequenceDataset

    from rnagym.fitness.baselines.GenSLM import compute_fitness as scorer

    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    load = genslm.GenSLM

    def capture_model(name, model_cache_dir):
        assert name == "genslm_2.5B_patric"
        assert Path(model_cache_dir) == ConfigFitness.CHECKPOINT_DIR / "genslm/2.5B"
        model = load(name, model_cache_dir=model_cache_dir)
        models.append(model)
        return model

    monkeypatch.setattr(genslm, "GenSLM", capture_model)
    monkeypatch.setattr(ConfigFitness, "ASSAY_DIR", FIXTURE_DIR / "assays")
    monkeypatch.setattr(ConfigFitness, "REFERENCE_FILE", FIXTURE_DIR / "reference.csv")
    args = scorer.parse_args(shlex.split(f"--rows 2 --output {tmp_path}"))
    scorer.main(args)
    assert len(models) == 1
    model = models[0]
    assay = pl.read_csv(FIXTURE_DIR / "assays/Tome_2014_GFP_aptamer.csv")
    result = pl.read_csv(tmp_path / "Tome_2014_GFP_aptamer.csv")
    assert_frame_equal(result.select(assay.columns), assay)
    assert result["logit_scores"].is_finite().all()
    assert (result["logit_scores"] < 0).all()

    def native_score(sequence):
        encoded = SequenceDataset(
            [sequence], (len(sequence) + 2) // 3, model.tokenizer, verbose=False
        )[0]
        assert encoded["attention_mask"].all()
        with torch.inference_mode():
            return -model(
                encoded["input_ids"][None].to(device),
                encoded["attention_mask"].to(device),
            ).loss.item()

    rows = np.linspace(0, assay.height - 1, 8, dtype=int).tolist()
    expected = [native_score(s.replace("U", "T")) for s in assay[rows]["sequence"]]
    np.testing.assert_allclose(
        result[rows]["logit_scores"], expected, atol=1e-5, rtol=1e-6
    )

    # Different real construct lengths test padding and batch independence
    sequences = [
        sequence.replace("U", "T")
        for path in sorted((FIXTURE_DIR / "assays").glob("*.csv"))
        for sequence in pl.read_csv(path)["sequence"].head(2)
    ]
    expected = [native_score(sequence) for sequence in sequences]
    for batch_size in (1, 4):
        actual = scorer.sequence_log_likelihoods(model, sequences, device, batch_size)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
