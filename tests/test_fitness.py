"""Reproduce the fitness workflow on complete released assays."""

import json
import runpy
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fitness import analyze_fill_strategies
from fitness.baselines.masked_lm import (
    MASK_CHAR,
    MaskedLMAdapter,
    accumulate_scores,
    build_tasks,
    parse_mutations,
    runner,
    window_contexts,
)
from fitness.baselines.masked_lm.strategies import validate_table
from fitness.merge_scoring_files import combine_csv_data
from fitness.model_registry import ALL_MODELS, SCORE_COLS, resolve_source
from fitness.performance_fitness import calculate_metrics, get_performance_dataset

REPOSITORY = Path(__file__).parent.parent
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fitness"
ASSAY_DIR = FIXTURE_DIR / "assays"
PREDICTION_DIR = FIXTURE_DIR / "predictions" / "rna_fm_4fill"
REFERENCE_FILE = FIXTURE_DIR / "reference.csv"
ASSAY_NAMES = (
    "Andreasson_2020_ribozyme.csv",
    "Domingo_2018_tRNA.csv",
    "Tome_2014_GFP_aptamer.csv",
)
EXPECTED_FILES = (
    "results_by_rna_type.csv",
    "results_by_mutation_depth.csv",
    "results_by_rna_type_and_depth.csv",
    "assay_level_results.csv",
    "assay_level_results_transposed.csv",
    "fill_strategy_per_assay.csv",
    "fill_strategy_macro.csv",
)
MODEL_NAMES = tuple(f"rna_fm_{strategy}" for strategy in runner.STRATEGIES)
PREDICTION_COLUMNS = {
    f"rna_fm_{strategy}_score": f"RNA_FM_scores_{strategy}"
    for strategy in runner.STRATEGIES
}
VOCAB = {
    token: index
    for index, token in enumerate(
        ("<pad>", "<cls>", "<eos>", "<mask>", "A", "C", "G", "U", "N")
    )
}


class FixtureAdapter(MaskedLMAdapter):
    """Checkpoint-free adapter whose logits depend on the complete context."""

    name = "fixture masked LM"
    bases = "ACGU"
    n_special_tokens = 2
    score_column = "RNA_FM_scores"

    def __init__(self):
        self.base_ids = {base: VOCAB[base] for base in self.bases}
        self.device = "cpu"
        self.mask_id = VOCAB["<mask>"]
        self.pad_id = VOCAB["<pad>"]
        self.prefix_ids = [VOCAB["<cls>"]]
        self.suffix_ids = [VOCAB["<eos>"]]
        self.unk_id = VOCAB["N"]

    @staticmethod
    def add_arguments(parser):
        """The fixture has no model-specific arguments."""

    def load(self, args):
        """Use the device selected by the shared runner."""
        self.device = args.device

    def logits_at(self, input_ids, attention_mask, rows, cols):
        """Return deterministic context- and base-dependent logits."""
        positions = torch.arange(1, input_ids.shape[1] + 1, device=self.device)
        context = (input_ids * positions).sum(dim=1)[rows, None]
        vocab = torch.arange(len(VOCAB), device=self.device)[None, :]
        logits = context * (vocab + 1) + cols[:, None] * (vocab + 2)
        return torch.remainder(logits, 23).float() / 4


class NonfiniteAdapter(FixtureAdapter):
    """Adapter that simulates a numerically failed model forward pass."""

    def logits_at(self, input_ids, attention_mask, rows, cols):
        logits = super().logits_at(input_ids, attention_mask, rows, cols)
        return torch.full_like(logits, torch.nan)


class LimitedFixtureAdapter(FixtureAdapter):
    """Fixture adapter with a deliberately small position limit."""

    max_tokens = 20


def run_cli(monkeypatch, command):
    """Run a Python command line entry point in the current process."""
    arguments = shlex.split(command)
    script = Path(arguments[0]).resolve()
    arguments[0] = str(script)
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, "argv", arguments)
    runpy.run_path(str(script), run_name="__main__")


def fixture_log_probs(adapter, context, position):
    """Score one masked position without the shared batching engine."""
    input_ids = torch.tensor([adapter.encode_context(context, MASK_CHAR)])
    logits = adapter.logits_at(
        input_ids,
        torch.ones_like(input_ids),
        torch.tensor([0]),
        torch.tensor([adapter.token_position(position)]),
    )
    return torch.log_softmax(logits.float(), dim=-1)[0].numpy()


def masked(sequence, positions):
    """Replace selected sequence positions with the fixture mask."""
    context = list(sequence)
    for position in positions:
        context[position] = MASK_CHAR
    return "".join(context)


def runner_command(assay_dir, output_dir):
    """Build the shared runner command for the Domingo fixture."""
    return (
        f"fitness-fixture --row_id 1 --ref_sheet {REFERENCE_FILE} "
        f"--dms_dir_path {assay_dir} --output_dir_path {output_dir} "
        "--device cpu --quiet_skips"
    )


def test_fitness_workflow(tmp_path, monkeypatch):
    """Reproduce the benchmark workflow from released RNA-FM predictions."""
    merged_dir = tmp_path / "merged"
    performance_dir = tmp_path / "performance"
    analysis_dir = tmp_path / "analysis"
    merged_dir.mkdir()
    (merged_dir / "stale.csv").write_text("stale\n")

    models = " ".join(MODEL_NAMES)
    run_cli(
        monkeypatch,
        f"fitness/merge_scoring_files.py --processed_folder {ASSAY_DIR} "
        f"--model_predictions_folder {PREDICTION_DIR.parent} "
        f"--output_folder {merged_dir} --reference_file {REFERENCE_FILE} "
        f"--models {models}",
    )
    assert not (merged_dir / "stale.csv").exists()
    run_cli(
        monkeypatch,
        f"fitness/performance_fitness.py --reference_file {REFERENCE_FILE} "
        f"--combined_dir {merged_dir} --performance_dir {performance_dir} "
        f"--type ncRNA --models {models}",
    )

    for assay_name in ASSAY_NAMES:
        assay = (
            pd.read_csv(ASSAY_DIR / assay_name)
            .dropna(subset=["mutant"])
            .reset_index(drop=True)
        )
        prediction = (
            pd.read_csv(PREDICTION_DIR / assay_name)
            .dropna(subset=["mutant"])
            .reset_index(drop=True)
        )
        merged = pd.read_csv(merged_dir / assay_name)
        pd.testing.assert_frame_equal(merged[assay.columns], assay, check_exact=True)
        for merged_column, prediction_column in PREDICTION_COLUMNS.items():
            np.testing.assert_allclose(
                merged[merged_column], prediction[prediction_column], rtol=0, atol=1e-14
            )
        score_columns = list(PREDICTION_COLUMNS.values())
        single = ~prediction["mutant"].str.contains(",")
        assert np.allclose(
            prediction.loc[single, score_columns],
            prediction.loc[single, score_columns[0]].to_numpy()[:, None],
        )
        if (~single).any():
            assert np.count_nonzero(
                np.ptp(prediction.loc[~single, score_columns].to_numpy(), axis=1)
            )

    run_cli(
        monkeypatch,
        f"{analyze_fill_strategies.__file__} "
        f"--predictions_folder {PREDICTION_DIR.parent} "
        f"--ref_sheet {REFERENCE_FILE} --output_folder {analysis_dir}",
    )
    for result_file in EXPECTED_FILES:
        result_dir = (
            analysis_dir if result_file.startswith("fill_strategy") else performance_dir
        )
        observed = pd.read_csv(result_dir / result_file)
        expected = pd.read_csv(FIXTURE_DIR / "expected" / result_file)
        pd.testing.assert_frame_equal(
            observed, expected, check_exact=False, rtol=0, atol=1e-10
        )

    masked_models = [
        model for model in ALL_MODELS if isinstance(SCORE_COLS[model], dict)
    ]
    for model in masked_models:
        folder, column = resolve_source(SCORE_COLS, model)
        assert folder.endswith("_4fill")
        assert column.endswith("_wt_fill")
    leaderboard_models = set(
        pd.read_csv(
            REPOSITORY / "leaderboard" / "fitness" / "leaderboard_signed_3ncRNA.csv"
        )["model"]
    )
    assert leaderboard_models <= set(SCORE_COLS)


def test_masked_lm_scoring_and_guards(tmp_path, monkeypatch):
    """Score a complete real assay and reject corrupt scoring states."""
    assay = pd.read_csv(FIXTURE_DIR / "assays" / "Domingo_2018_tRNA.csv", nrows=1)
    reference = pd.read_csv(FIXTURE_DIR / "reference.csv").set_index("DMS_ID")
    wild_type = reference.loc["Domingo_2018_tRNA", "RAW_CONSTRUCT_SEQ"]
    adapter = NonfiniteAdapter()
    table = build_tasks(
        assay["mutant"],
        assay["sequence"],
        wild_type,
        adapter.bases,
        ("wt_fill",),
        verbose=False,
    )
    position = table.pos[0]
    table.pos[0] = len(table.contexts[table.ctx_id[0]])
    with pytest.raises(ValueError, match="falls outside"):
        validate_table(table)
    table.pos[0] = position

    with pytest.raises(FloatingPointError, match="non-finite log probability"):
        accumulate_scores(adapter, table, 1, 1, 256, progress=False)

    with pytest.raises(ValueError, match="cannot hold one encoded context"):
        accumulate_scores(FixtureAdapter(), table, 1, 1, 1, progress=False)

    window_contexts(table, budget=20)
    assert table.contexts[0][table.pos[0]] == MASK_CHAR
    assert np.isfinite(
        accumulate_scores(FixtureAdapter(), table, 1, 1, 64, progress=False)
    ).all()

    variant = pd.read_csv(FIXTURE_DIR / "assays" / "Domingo_2018_tRNA.csv").iloc[-1]
    mutations = parse_mutations(variant["mutant"], FixtureAdapter.bases)
    formula_table = build_tasks(
        [variant["mutant"]],
        [variant["sequence"]],
        wild_type,
        FixtureAdapter.bases,
        runner.STRATEGIES,
        verbose=False,
    )
    formula_scores = accumulate_scores(
        FixtureAdapter(), formula_table, 1, 16, 4096, progress=False
    )[:, 0]
    expected = np.zeros(len(runner.STRATEGIES))
    positions = [position for position, _, _ in mutations]
    joint_context = masked(wild_type, positions)
    fixture_adapter = FixtureAdapter()
    for position, wild_base, mutant_base in mutations:
        wild_context = masked(wild_type, [position])
        mutant_context = masked(variant["sequence"], [position])
        wild_probs = fixture_log_probs(fixture_adapter, wild_context, position)
        joint_probs = fixture_log_probs(fixture_adapter, joint_context, position)
        mutant_probs = fixture_log_probs(fixture_adapter, mutant_context, position)
        wild_id = fixture_adapter.base_ids[wild_base]
        mutant_id = fixture_adapter.base_ids[mutant_base]
        expected += [
            wild_probs[mutant_id] - wild_probs[wild_id],
            joint_probs[mutant_id] - joint_probs[wild_id],
            mutant_probs[mutant_id] - mutant_probs[wild_id],
            mutant_probs[mutant_id] - wild_probs[wild_id],
        ]
    assert formula_scores == pytest.approx(expected, abs=1e-6)

    nan_table = build_tasks(
        [None, variant["mutant"]],
        [wild_type, variant["sequence"]],
        wild_type,
        FixtureAdapter.bases,
        ("wt_fill",),
        verbose=False,
    )
    nan_scores = accumulate_scores(
        FixtureAdapter(), nan_table, 2, 16, 4096, progress=False
    )
    assert np.isnan(nan_scores[0, 0])
    assert np.isfinite(nan_scores[0, 1])

    assay_dir = tmp_path / "assays"
    output_dir = tmp_path / "predictions"
    assay_dir.mkdir()
    shutil.copy2(
        FIXTURE_DIR / "assays" / "Domingo_2018_tRNA.csv",
        assay_dir / "Domingo_2018_tRNA.csv",
    )
    monkeypatch.setattr(sys, "argv", shlex.split(runner_command(assay_dir, output_dir)))

    runner.main(FixtureAdapter())
    scored = pd.read_csv(output_dir / "Domingo_2018_tRNA.csv")
    score_columns = [f"RNA_FM_scores_{strategy}" for strategy in runner.STRATEGIES]
    assert np.isfinite(scored[score_columns]).all().all()
    single = ~scored["mutant"].str.contains(",")
    assert np.allclose(
        scored.loc[single, score_columns],
        scored.loc[single, score_columns[0]].to_numpy()[:, None],
    )
    assert (
        np.count_nonzero(np.ptp(scored.loc[~single, score_columns].to_numpy(), axis=1))
        > 3000
    )
    manifest = json.loads((output_dir / "Domingo_2018_tRNA.manifest.json").read_text())
    assert manifest["scorable_variants"] == len(scored)
    assert manifest["contexts"] == 19275
    assert manifest["environment"]["device"] == "cpu"
    assert manifest["terms"] == 185632
    assert manifest["strategies"] == [
        "wt-fill",
        "mask-fill",
        "mut-fill",
        "match-fill",
    ]

    invalid_output_dir = tmp_path / "invalid_predictions"
    monkeypatch.setattr(
        sys, "argv", shlex.split(runner_command(assay_dir, invalid_output_dir))
    )
    with pytest.raises(SystemExit):
        runner.main(LimitedFixtureAdapter())
    assert not list(invalid_output_dir.glob("*.csv"))

    def inconsistent_scores(adapter, table, n_rows, **kwargs):
        scores = np.tile(np.arange(n_rows), (len(table.strategies), 1)).astype(float)
        scores[1] += 1.0
        return scores

    single_assay_dir = tmp_path / "single_assay"
    single_assay_dir.mkdir()
    domingo = pd.read_csv(FIXTURE_DIR / "assays" / "Domingo_2018_tRNA.csv")
    domingo[~domingo["mutant"].str.contains(",")].to_csv(
        single_assay_dir / "Domingo_2018_tRNA.csv", index=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        shlex.split(runner_command(single_assay_dir, invalid_output_dir)),
    )
    monkeypatch.setattr(runner, "accumulate_scores", inconsistent_scores)
    with pytest.raises(SystemExit):
        runner.main(FixtureAdapter())
    assert not list(invalid_output_dir.glob("*.csv"))
    assert not list(invalid_output_dir.glob("*.manifest.json"))


def test_metrics_omit_nonfinite_pairs():
    """Treat infinities like missing predictions instead of losing an assay."""
    frame = pd.DataFrame(
        {
            "mutant": ["A1C"] * 6,
            "DMS_score": np.arange(6, dtype=float),
            "model_score": [0.0, 1.0, np.inf, 3.0, np.nan, 5.0],
        }
    )
    observed = get_performance_dataset(frame, "DMS_score", ["model_score"])[
        "model_score"
    ]
    expected = calculate_metrics(
        np.array([0.0, 1.0, 3.0, 5.0]), np.array([0.0, 1.0, 3.0, 5.0])
    )
    assert observed == pytest.approx(expected)


def test_merge_rejects_incomplete_predictions(tmp_path):
    """Reject missing assays and partial rows before replacing merged output."""
    assay_name = "Domingo_2018_tRNA.csv"
    assay_dir = tmp_path / "assays"
    prediction_dir = tmp_path / "predictions" / "rna_fm_4fill"
    assay_dir.mkdir()
    prediction_dir.mkdir(parents=True)
    shutil.copy2(FIXTURE_DIR / "assays" / assay_name, assay_dir / assay_name)
    merged_dir = tmp_path / "merged"

    with pytest.raises(FileNotFoundError, match="Prediction coverage is incomplete"):
        combine_csv_data(
            assay_dir,
            prediction_dir.parent,
            merged_dir,
            ["RNA-FM"],
            SCORE_COLS,
        )
    assert not merged_dir.exists()

    predictions = pd.read_csv(
        FIXTURE_DIR / "predictions" / "rna_fm_4fill" / assay_name
    ).iloc[:-1]
    predictions.to_csv(prediction_dir / assay_name, index=False)

    with pytest.raises(ValueError, match="Row count mismatch"):
        combine_csv_data(
            assay_dir,
            prediction_dir.parent,
            merged_dir,
            ["RNA-FM"],
            SCORE_COLS,
        )
    assert not merged_dir.exists()
