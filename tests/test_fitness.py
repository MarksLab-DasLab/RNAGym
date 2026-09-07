"""Reproduce the fitness workflow on complete released assays."""

import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from rnagym.config import ConfigFitness
from rnagym.fitness.baselines.Evo import score_evo2_single_dms as evo2
from rnagym.fitness.baselines.masked_lm import (
    MASK_CHAR,
    MaskedLMAdapter,
    accumulate_scores,
    build_tasks,
    parse_mutations,
    runner,
    window_contexts,
)
from rnagym.fitness.baselines.masked_lm.engine import pad_contexts
from rnagym.fitness.baselines.masked_lm.strategies import validate_table
from rnagym.fitness.baselines.Nucleotide_Transformer.compute_fitness import NTv3Adapter
from rnagym.fitness.tasks import (
    analyze_fill_strategies,
    check_published,
    merge_scoring_files,
    performance_fitness,
)
from rnagym.fitness.tasks.merge_scoring_files import combine_csv_data
from rnagym.fitness.tasks.model_registry import (
    ALL_MODELS,
    SCORE_COLS,
    resolve_source,
)
from rnagym.fitness.tasks.performance_fitness import get_performance_dataset

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
        self.loads = getattr(self, "loads", 0) + 1

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


class PaddedFixtureAdapter(FixtureAdapter):
    """Fixture adapter that requires 128-position contexts."""

    context_pad_char = "N"

    def context_length_for(self, length):
        return 128


def run_cli(entry_point, command):
    """Run a package command line entry point in the current process."""
    entry_point(shlex.split(command))


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


def runner_command(output_dir):
    """Build the shared runner command for the Domingo fixture."""
    return f"fitness-fixture --rows 1 --output {output_dir}"


@pytest.mark.filterwarnings("ignore:An input array is constant")
@pytest.mark.filterwarnings("ignore:Only one class is present in y_true")
def test_fitness_workflow(tmp_path, monkeypatch):
    """Reproduce the benchmark workflow from released RNA-FM predictions."""
    assert (
        ConfigFitness.REFERENCE_FILE
        == REPOSITORY / "data" / "fitness" / "reference_sheet_final.csv"
    )

    monkeypatch.setattr(ConfigFitness, "ASSAY_DIR", ASSAY_DIR)
    monkeypatch.setattr(ConfigFitness, "PREDICTION_DIR", PREDICTION_DIR.parent)
    monkeypatch.setattr(ConfigFitness, "REFERENCE_FILE", REFERENCE_FILE)

    merged_dir = tmp_path / "merged"
    performance_dir = tmp_path / "performance"
    analysis_dir = tmp_path / "analysis"
    merged_dir.mkdir()
    (merged_dir / "stale.csv").write_text("stale\n")

    models = " ".join(MODEL_NAMES)
    run_cli(
        merge_scoring_files.main,
        f"--output {merged_dir} --models {models}",
    )
    assert not (merged_dir / "stale.csv").exists()
    run_cli(
        performance_fitness.cli,
        f"--input {merged_dir} --output {performance_dir} --models {models}",
    )

    for assay_name in ASSAY_NAMES:
        assay = pl.read_csv(ASSAY_DIR / assay_name).drop_nulls("mutant")
        prediction = pl.read_csv(PREDICTION_DIR / assay_name).drop_nulls("mutant")
        merged = pl.read_csv(merged_dir / assay_name)
        assert_frame_equal(merged.select(assay.columns), assay, check_exact=True)
        for merged_column, prediction_column in PREDICTION_COLUMNS.items():
            np.testing.assert_allclose(
                merged[merged_column], prediction[prediction_column], rtol=0, atol=1e-14
            )
        score_columns = list(PREDICTION_COLUMNS.values())
        single = ~prediction["mutant"].str.contains(",")
        assert np.allclose(
            prediction.filter(single).select(score_columns).to_numpy(),
            prediction.filter(single)[score_columns[0]].to_numpy()[:, None],
        )
        if (~single).any():
            assert np.count_nonzero(
                np.ptp(
                    prediction.filter(~single).select(score_columns).to_numpy(), axis=1
                )
            )

    run_cli(
        analyze_fill_strategies.main,
        f"--output {analysis_dir}",
    )
    for result_file in EXPECTED_FILES:
        result_dir = (
            analysis_dir if result_file.startswith("fill_strategy") else performance_dir
        )
        observed = pl.read_csv(result_dir / result_file)
        expected = pl.read_csv(FIXTURE_DIR / "expected" / result_file)
        assert_frame_equal(
            observed, expected, check_exact=False, rel_tol=0, abs_tol=1e-10
        )

    masked_models = [
        model for model in ALL_MODELS if isinstance(SCORE_COLS[model], dict)
    ]
    for model in masked_models:
        folder, column = resolve_source(SCORE_COLS, model)
        assert folder.endswith("_4fill")
        assert column.endswith("_wt_fill")
    leaderboard_models = set(
        pl.read_csv(
            REPOSITORY / "leaderboard" / "fitness" / "leaderboard_signed_3ncRNA.csv"
        )["model"]
    )
    assert leaderboard_models <= set(SCORE_COLS)

    # Undefined assay correlations must not disappear from the category average
    assay_file = merged_dir / ASSAY_NAMES[0]
    original = pl.read_csv(assay_file)
    saved_results = {path: path.read_bytes() for path in performance_dir.glob("*.csv")}
    for invalid in (
        original.with_columns(pl.lit(0.0).alias(next(iter(PREDICTION_COLUMNS)))),
        original.with_columns(pl.lit(0.0).alias("DMS_score")),
        original.head(0),
    ):
        invalid.write_csv(assay_file)
        with pytest.raises(ValueError, match="undefined Spearman correlation"):
            run_cli(
                performance_fitness.cli,
                f"--input {merged_dir} --output {performance_dir} --models {models}",
            )
        assert all(
            path.read_bytes() == content for path, content in saved_results.items()
        )
    original.write_csv(assay_file)


def test_masked_lm_scoring_and_guards(tmp_path, monkeypatch):
    """Score a complete real assay and reject corrupt scoring states."""
    assay = pl.read_csv(FIXTURE_DIR / "assays" / "Domingo_2018_tRNA.csv", n_rows=1)
    reference = pl.read_csv(FIXTURE_DIR / "reference.csv")
    wild_type = reference.filter(pl.col("DMS_ID") == "Domingo_2018_tRNA")[
        "RAW_CONSTRUCT_SEQ"
    ][0]
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

    variant = pl.read_csv(FIXTURE_DIR / "assays" / "Domingo_2018_tRNA.csv").row(
        -1, named=True
    )
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

    padded_table = build_tasks(
        [variant["mutant"]],
        [variant["sequence"]],
        wild_type,
        PaddedFixtureAdapter.bases,
        ("wt_fill",),
        verbose=False,
    )
    positions = padded_table.pos.copy()
    pad_contexts(padded_table, PaddedFixtureAdapter())
    assert {len(context) for context in padded_table.contexts} == {128}
    assert np.array_equal(padded_table.pos, positions)
    assert all(
        context.endswith("N" * (128 - len(wild_type)))
        for context in padded_table.contexts
    )

    ntv3 = NTv3Adapter()
    ntv3.num_downsamples = 7
    assert [ntv3.context_length_for(n) for n in (45, 128, 129, 425)] == [
        128,
        128,
        256,
        512,
    ]

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
    monkeypatch.setattr(ConfigFitness, "ASSAY_DIR", assay_dir)
    monkeypatch.setattr(ConfigFitness, "REFERENCE_FILE", REFERENCE_FILE)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", shlex.split(runner_command(output_dir)))

    shutil.copy2(ASSAY_DIR / ASSAY_NAMES[2], assay_dir / ASSAY_NAMES[2])
    monkeypatch.setattr(
        sys,
        "argv",
        shlex.split(runner_command(output_dir).replace("--rows 1", "--rows 1-2")),
    )
    adapter = FixtureAdapter()
    runner.main(adapter)
    assert adapter.loads == 1
    assert pl.read_csv(output_dir / ASSAY_NAMES[2]).height == 417
    scored = pl.read_csv(output_dir / "Domingo_2018_tRNA.csv")
    score_columns = [f"RNA_FM_scores_{strategy}" for strategy in runner.STRATEGIES]
    assert np.isfinite(scored.select(score_columns).to_numpy()).all()
    single = ~scored["mutant"].str.contains(",")
    assert np.allclose(
        scored.filter(single).select(score_columns).to_numpy(),
        scored.filter(single)[score_columns[0]].to_numpy()[:, None],
    )
    assert (
        np.count_nonzero(
            np.ptp(scored.filter(~single).select(score_columns).to_numpy(), axis=1)
        )
        > 3000
    )
    manifest = json.loads((output_dir / "Domingo_2018_tRNA.manifest.json").read_text())
    assert (
        manifest["prediction_sha256"]
        == hashlib.sha256(
            (output_dir / "Domingo_2018_tRNA.csv").read_bytes()
        ).hexdigest()
    )
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
    monkeypatch.setattr(sys, "argv", shlex.split(runner_command(invalid_output_dir)))
    with pytest.raises(SystemExit):
        runner.main(LimitedFixtureAdapter())
    assert not list(invalid_output_dir.glob("*.csv"))

    def inconsistent_scores(adapter, table, n_rows, **kwargs):
        scores = np.tile(np.arange(n_rows), (len(table.strategies), 1)).astype(float)
        scores[1] += 1.0
        return scores

    single_assay_dir = tmp_path / "single_assay"
    single_assay_dir.mkdir()
    domingo = pl.read_csv(FIXTURE_DIR / "assays" / "Domingo_2018_tRNA.csv")
    domingo.filter(~pl.col("mutant").str.contains(",")).write_csv(
        single_assay_dir / "Domingo_2018_tRNA.csv"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        shlex.split(runner_command(invalid_output_dir)),
    )
    monkeypatch.setattr(ConfigFitness, "ASSAY_DIR", single_assay_dir)
    monkeypatch.setattr(runner, "accumulate_scores", inconsistent_scores)
    with pytest.raises(SystemExit):
        runner.main(FixtureAdapter())
    assert not list(invalid_output_dir.glob("*.csv"))
    assert not list(invalid_output_dir.glob("*.manifest.json"))


def test_metrics_reject_nonfinite_predictions():
    """Reject incomplete predictions instead of changing the evaluation set."""
    frame = pl.DataFrame(
        {
            "mutant": ["A1C"] * 6,
            "DMS_score": np.arange(6, dtype=float),
            "model_score": [0.0, 1.0, np.inf, 3.0, np.nan, 5.0],
        }
    )
    with pytest.raises(ValueError, match="model_score"):
        get_performance_dataset(frame, "DMS_score", ["model_score"])
    with pytest.raises(KeyError, match="missing_score"):
        get_performance_dataset(frame, "DMS_score", ["missing_score"])


def test_published_comparison_tolerance(tmp_path):
    """Accept GPU rounding drift while preserving prediction structure."""
    import polars as pl

    expected_file = tmp_path / "expected.csv"
    actual_file = tmp_path / "actual.csv"
    expected = pl.DataFrame(
        {"mutant": ["A1C", "C2G"], "sequence": ["CC", "AG"], "score": [0.0, None]}
    )
    expected.write_csv(expected_file)

    exact = check_published.compare_predictions(
        expected_file, expected_file, ("score",)
    )
    assert exact.exact and exact.max_absolute_difference == 0.0

    close_score = check_published.ABSOLUTE_TOLERANCE * 0.98
    expected.with_columns(pl.Series("score", [close_score, None])).write_csv(
        actual_file
    )
    close = check_published.compare_predictions(actual_file, expected_file, ("score",))
    assert not close.exact
    assert close.max_absolute_difference == pytest.approx(close_score)

    rejected_score = check_published.ABSOLUTE_TOLERANCE * 1.02
    expected.with_columns(pl.Series("score", [rejected_score, None])).write_csv(
        actual_file
    )
    with pytest.raises(AssertionError, match="absolute difference"):
        check_published.compare_predictions(actual_file, expected_file, ("score",))

    expected.with_columns(pl.Series("score", [0.0, 1.0])).write_csv(actual_file)
    with pytest.raises(AssertionError, match="missing-value status"):
        check_published.compare_predictions(actual_file, expected_file, ("score",))


def test_evo2_workflow_and_guards(tmp_path, monkeypatch):
    """Score complete assays in one model load and reject failed inference."""

    class FixtureEvo2:
        calls = []
        fp8 = True
        loads = 0
        nonfinite = False

        def __init__(self, model_name, local_path=None):
            type(self).loads += 1
            self.model = SimpleNamespace(
                config={"use_fp8_input_projections": type(self).fp8}
            )

        def score_sequences(self, sequences, **kwargs):
            type(self).calls.append((sequences[0], kwargs))
            scores = np.array(
                [sequence.count("T") / len(sequence) for sequence in sequences]
            )
            if type(self).nonfinite:
                scores[0] = np.nan
            return scores

    assay_dir = tmp_path / "assays"
    shutil.copytree(ASSAY_DIR, assay_dir)
    domingo_file = assay_dir / "Domingo_2018_tRNA.csv"
    output_dir = tmp_path / "evo2"
    monkeypatch.setattr(ConfigFitness, "ASSAY_DIR", assay_dir)
    monkeypatch.setattr(ConfigFitness, "REFERENCE_FILE", REFERENCE_FILE)
    command = f"--rows 0-2 --output {output_dir} --model evo2_1b_base"
    args = evo2.parse_args(shlex.split(command))

    evo2.run(args, FixtureEvo2)
    assert FixtureEvo2.loads == 1
    assert len(FixtureEvo2.calls) == 3
    for sequence, options in FixtureEvo2.calls:
        assert options == {
            "batch_size": max(
                1, 32768 // evo2.effective_length(len(sequence), False, True)
            ),
            "prepend_bos": False,
            "reduce_method": "mean",
            "average_reverse_complement": True,
        }
    for assay_file in sorted(assay_dir.glob("*.csv")):
        output_file = output_dir / assay_file.name
        observed = pl.read_csv(output_file)
        source = pl.read_csv(assay_file)
        score_column = "evo2_1b_base_score"
        expected = (
            source["sequence"]
            .str.to_uppercase()
            .str.replace_all("U", "T")
            .str.count_matches("T")
            / source["sequence"].str.len_chars()
        )
        np.testing.assert_allclose(observed[score_column], expected)
        assert output_file.stat().st_mode & 0o777 == 0o644
    assert not list(output_dir.glob("*.tmp"))

    evo2.run(args, FixtureEvo2)
    assert FixtureEvo2.loads == 2
    args.rows = "0"
    args.output = tmp_path / "invalid_evo2"
    FixtureEvo2.fp8 = False
    with pytest.raises(RuntimeError, match="does not use FP8"):
        evo2.run(args, FixtureEvo2)

    FixtureEvo2.fp8 = True
    FixtureEvo2.nonfinite = True
    with pytest.raises(FloatingPointError, match="nonfinite model scores"):
        evo2.run(args, FixtureEvo2)
    assert not list(args.output.glob("*.csv"))

    FixtureEvo2.nonfinite = False
    domingo = pl.read_csv(domingo_file)
    domingo = (
        domingo.with_row_index()
        .with_columns(
            pl.when(pl.col("index") == 0)
            .then(None)
            .otherwise(pl.col("sequence"))
            .alias("sequence")
        )
        .drop("index")
    )
    domingo.write_csv(domingo_file)
    args.rows = "1"
    with pytest.raises(ValueError, match="missing or empty sequences"):
        evo2.run(args, FixtureEvo2)


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

    predictions = pl.read_csv(
        FIXTURE_DIR / "predictions" / "rna_fm_4fill" / assay_name
    ).head(-1)
    predictions.write_csv(prediction_dir / assay_name)

    with pytest.raises(ValueError, match="Row count mismatch"):
        combine_csv_data(
            assay_dir,
            prediction_dir.parent,
            merged_dir,
            ["RNA-FM"],
            SCORE_COLS,
        )
    assert not merged_dir.exists()
