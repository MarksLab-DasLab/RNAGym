"""Reproduce the fitness workflow on complete released assays."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
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
    CHECKPOINT_REVISIONS,
    SCORE_COLS,
    checkpoint_revision,
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


def test_checkpoint_download(tmp_path, monkeypatch):
    for model in CHECKPOINT_REVISIONS:
        revision = checkpoint_revision(model)
        assert revision is not None and len(revision) == 40
        assert set(revision) <= set("0123456789abcdef")
    with pytest.raises(ValueError, match="Unpinned checkpoint"):
        checkpoint_revision("unregistered/model")
    local_model = tmp_path / "local checkpoint"
    local_model.mkdir()
    assert checkpoint_revision(str(local_model)) is None

    source = ASSAY_DIR / ASSAY_NAMES[0]
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "shared checkpoints" / "model" / "weights.csv"
    helper = REPOSITORY / "rnagym/sh/weights.sh"
    shell = 'source "$1" && download "$2" "$3" "$4"'
    command = (
        f"bash -c {shlex.quote(shell)} checkpoint-download {shlex.quote(str(helper))} "
        f"{shlex.quote(source.resolve().as_uri())} {shlex.quote(str(destination))} {checksum}"
    )
    environment = {**os.environ, "RNAGYM_CHECKPOINT_DIR": str(destination.parents[1])}
    with subprocess.Popen(shlex.split(command), env=environment) as first:
        with subprocess.Popen(shlex.split(command), env=environment) as second:
            assert first.wait(timeout=20) == second.wait(timeout=20) == 0
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o044 == 0o044
    failed = subprocess.run(
        shlex.split(command.replace(checksum, "0" * 64)),
        env=environment,
        check=False,
        timeout=20,
    )
    assert failed.returncode != 0
    assert destination.read_bytes() == source.read_bytes()
    assert list(destination.parent.iterdir()) == [destination]

    from rnagym.fitness.baselines.Evo import checkpoints

    content = source.read_bytes()
    split = len(content) // 2
    parts = [tmp_path / f"part{index}" for index in range(2)]
    for path, data in zip(parts, (content[:split], content[split:])):
        path.write_bytes(data)
    monkeypatch.setattr(
        checkpoints,
        "EVO2_40B_PARTS",
        tuple(
            (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in parts
        ),
    )
    monkeypatch.setattr(ConfigFitness, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    downloads = []

    def download_part(repo_id, filename, revision):
        assert repo_id == "arcinstitute/evo2_40b"
        assert revision == "d529aa57c30771814217ad89baaeaf6e2315c7d7"
        downloads.append(filename)
        return str(parts[int(filename.rsplit("part", 1)[1])])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=download_part),
    )
    merged = Path(checkpoints.evo2_checkpoint("evo2_40b"))
    assert merged.read_bytes() == content
    assert downloads == ["evo2_40b.pt.part0", "evo2_40b.pt.part1"]
    assert checkpoints.evo2_checkpoint("evo2_40b") == str(merged)
    assert len(downloads) == 2
    for corrupted in (b"X" + content[1:], content[:-1], content + b"X"):
        merged.write_bytes(corrupted)
        with pytest.raises(ValueError, match="checksum|Truncated|trailing"):
            checkpoints.evo2_checkpoint("evo2_40b")
        assert merged.read_bytes() == corrupted
    merged.unlink()
    parts[1].unlink()
    with pytest.raises(FileNotFoundError):
        checkpoints.evo2_checkpoint("evo2_40b")
    assert list(merged.parent.iterdir()) == [merged.with_suffix(".lock")]


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
    """Reject structural errors, nonfinite scores and plausible-looking wrong rankings."""
    expected_file = PREDICTION_DIR / "Tome_2014_GFP_aptamer.csv"
    expected = pl.read_csv(expected_file)
    columns = tuple(PREDICTION_COLUMNS.values())
    actual_file = tmp_path / "actual.csv"
    exact = check_published.compare_predictions(expected_file, expected_file, columns)
    assert exact.exact and exact.max_absolute_difference == 0
    expected.with_columns(
        pl.col("mutant").str.replace_all("U", "T").str.replace_all(",", ", ")
    ).write_csv(actual_file)
    assert check_published.compare_predictions(
        actual_file, expected_file, columns
    ).exact
    shifted = expected.with_columns(pl.col(*columns) + 1e-5)
    shifted.write_csv(actual_file)
    close = check_published.compare_predictions(actual_file, expected_file, columns)
    assert not close.exact and close.minimum_correlation > 0.999
    assert close.max_absolute_difference == pytest.approx(1e-5)
    expected.with_columns(pl.col(*columns) * 2 + 10).write_csv(actual_file)
    rescaled = check_published.compare_predictions(actual_file, expected_file, columns)
    assert rescaled.minimum_correlation == pytest.approx(1.0)
    assert rescaled.max_metric_difference == 0
    shifted.write_csv(actual_file)
    lowercase = tmp_path / "lowercase.csv"
    expected.rename({"DMS_score": "dms_score"}).write_csv(lowercase)
    assert check_published.compare_predictions(actual_file, lowercase, columns) == close
    for corrupted in (
        expected.head(-1),
        expected.reverse(),
        expected.drop(columns[0]),
        expected.with_columns(pl.lit(float("inf")).alias(columns[0])),
        expected.with_columns(pl.lit(None, dtype=pl.Float64).alias(columns[0])),
        expected.with_columns(-pl.col(columns[0])),
        expected.with_columns(pl.lit(0.0).alias(columns[0])),
    ):
        corrupted.write_csv(actual_file)
        with pytest.raises((AssertionError, ValueError)):
            check_published.compare_predictions(actual_file, expected_file, columns)
    expected.drop("DMS_score").write_csv(lowercase)
    with pytest.raises(ValueError, match="measured fitness"):
        check_published.compare_predictions(expected_file, lowercase, columns)
    for scale in (0.0, 1e200):
        expected.with_columns(pl.col(*columns) * scale).write_csv(lowercase)
        with np.errstate(over="ignore", invalid="ignore"):
            with pytest.raises(AssertionError, match="no variation|normalized error"):
                check_published.compare_predictions(expected_file, lowercase, columns)

    # Small scores can pass an absolute tolerance despite losing all signal
    scaled_file = tmp_path / "scaled.csv"
    scaled = expected.with_columns(pl.col(*columns) * 1e-5)
    scaled.write_csv(scaled_file)
    scaled.with_columns(pl.lit(0.0).alias(columns[0])).write_csv(actual_file)
    with pytest.raises(AssertionError, match="normalized error"):
        check_published.compare_predictions(actual_file, scaled_file, columns)


def test_evo_precision_comparison(tmp_path):
    """Accept stable GPU rankings and reject the observed Evo1 BF16 failures."""
    fixture = pl.read_csv(FIXTURE_DIR / "evo.csv")
    actual_file, expected_file = tmp_path / "actual.csv", tmp_path / "expected.csv"
    comparisons = []
    for table in fixture.partition_by("model", "assay", maintain_order=True):
        for precision in ("float32", "bfloat16"):
            table.select(
                "mutant",
                "sequence",
                "DMS_score",
                pl.col(f"{precision}_h100").alias("score"),
            ).write_csv(expected_file)
            table.select(
                "mutant",
                "sequence",
                "DMS_score",
                pl.col(f"{precision}_l40s").alias("score"),
            ).write_csv(actual_file)
            if (
                precision == "bfloat16"
                and table["model"][0] == "evo1"
                and table["assay"][0] != "Domingo_2018_tRNA"
            ):
                with pytest.raises(AssertionError):
                    check_published.compare_predictions(
                        actual_file, expected_file, ("score",)
                    )
            else:
                comparison = check_published.compare_predictions(
                    actual_file, expected_file, ("score",)
                )
                if precision == "float32":
                    comparisons.append(comparison)
    assert len(comparisons) == 8 and sum(row.rows for row in comparisons) == 1972
    assert max(row.normalized_error for row in comparisons) < 0.001
    assert min(row.minimum_correlation for row in comparisons) > 0.9999
    assert max(row.max_metric_difference for row in comparisons) < 0.001


def test_fp8_comparison(tmp_path):
    """Accept measured Evo2 variation and reject rank agreement below 0.95."""
    fixture = pl.read_csv(FIXTURE_DIR / "fp8.csv")
    actual_file, expected_file = tmp_path / "actual.csv", tmp_path / "expected.csv"
    comparisons = []
    for table in fixture.partition_by("model", "assay", maintain_order=True):
        expected = table.select(
            "mutant", "sequence", "DMS_score", pl.col("published_score").alias("score")
        )
        expected.write_csv(expected_file)
        expected.with_columns(table["h100_score"].alias("score")).write_csv(actual_file)
        comparison = check_published.compare_predictions(
            actual_file, expected_file, ("score",)
        )
        assert not comparison.exact
        comparisons.append(comparison)
    assert len(comparisons) == 8 and sum(row.rows for row in comparisons) == 1972
    assert max(row.normalized_error for row in comparisons) == pytest.approx(0.2013094)
    assert min(row.minimum_correlation for row in comparisons) == pytest.approx(
        0.9874413
    )
    assert max(row.max_metric_difference for row in comparisons) < 0.005

    # Perturb real scores on either side of the accepted rank agreement
    for amplitude in (0.2, 0.4):
        expected.with_columns(
            pl.col("score")
            + amplitude * (pl.col("score").shuffle(seed=0) - pl.col("score").mean())
        ).write_csv(actual_file)
        if amplitude == 0.2:
            comparison = check_published.compare_predictions(
                actual_file, expected_file, ("score",)
            )
            assert comparison.minimum_correlation == pytest.approx(0.9776518)
        else:
            with pytest.raises(AssertionError, match="rank correlation"):
                check_published.compare_predictions(
                    actual_file, expected_file, ("score",)
                )

    # Fitness drift remains visible even when the score ranks agree
    measured = pl.col("DMS_score").rank()
    expected.with_columns(
        pl.col("score")
        + 0.1
        * pl.col("score").std(ddof=0)
        * (measured - measured.mean())
        / measured.std(ddof=0)
    ).write_csv(actual_file)
    comparison = check_published.compare_predictions(
        actual_file, expected_file, ("score",)
    )
    assert comparison.max_metric_difference > 0.01


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


def test_repeated_predictions(tmp_path):
    """Preserve repeated measurements and reproduce the published first-score convention."""
    from rnagym.fitness.tasks.merge_scoring_files import merge_predictions

    assay = pl.read_csv(FIXTURE_DIR / "repeated" / "assay.csv")
    prediction = pl.read_csv(FIXTURE_DIR / "repeated" / "prediction.csv")
    merged = merge_predictions(assay, prediction, "evo2_7b_score", "evo2")
    assert_frame_equal(merged.select(assay.columns), assay)
    assert merged.height == 186
    repeated = merged.filter(pl.col("mutant") == "A48C")
    assert repeated["evo2_score"].to_list() == [-1.1337192, -1.1337192]
    rho = get_performance_dataset(merged, "DMS_score", ["evo2_score"])["evo2_score"][
        "Spearman"
    ]
    assert rho == pytest.approx(-0.017426406587199662, abs=1e-14)
    corrupted = (
        prediction.with_row_index()
        .with_columns(
            pl.when(
                (pl.col("mutant") == "A48C")
                & (
                    pl.col("index")
                    == pl.col("index").filter(pl.col("mutant") == "A48C").max()
                )
            )
            .then(pl.col("evo2_7b_score") + 0.1)
            .otherwise(pl.col("evo2_7b_score"))
            .alias("evo2_7b_score")
        )
        .drop("index")
    )
    with pytest.raises(ValueError, match="conflicting scores"):
        merge_predictions(assay, corrupted, "evo2_7b_score", "evo2")
    with pytest.raises(ValueError, match="Row count mismatch"):
        merge_predictions(
            assay,
            prediction.filter(pl.col("mutant") != "A48C"),
            "evo2_7b_score",
            "evo2",
        )
    with pytest.raises(ValueError, match="sequence disagrees"):
        merge_predictions(
            assay,
            prediction.with_columns(pl.lit("ACGU").alias("sequence")),
            "evo2_7b_score",
            "evo2",
        )

    coding = pl.read_csv(FIXTURE_DIR / "repeated" / "coding.csv")
    assert coding.height == 106
    assays = tmp_path / "assays"
    predictions = tmp_path / "predictions" / "GenSLM"
    assays.mkdir()
    predictions.mkdir(parents=True)
    for (name,), group in coding.group_by("DMS_ID", maintain_order=True):
        group.select("mutant", "sequence", "DMS_score").write_csv(
            assays / f"{name}.csv"
        )
        group.drop("DMS_ID").write_csv(predictions / f"{name}.csv")
    combine_csv_data(
        assays, predictions.parent, tmp_path / "merged", ["GenSLM"], SCORE_COLS
    )
    for path in assays.glob("*.csv"):
        assay = pl.read_csv(path).with_columns(
            merge_scoring_files.standardize_mutation("mutant")
        )
        result = pl.read_csv(tmp_path / "merged" / path.name)
        assert_frame_equal(result.select(assay.columns), assay)
        assert (
            result.group_by("mutant")
            .agg(pl.col("GenSLM_score").n_unique())["GenSLM_score"]
            .eq(1)
            .all()
        )


def test_launchers_and_reproduction(tmp_path, monkeypatch):
    """Exercise real shell launchers and detect ignored row IDs in the reproduction harness."""
    import os
    import subprocess
    import time

    from rnagym.fitness.tasks.check_published import ModelFamily

    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    capture = tmp_path / "arguments.json"
    executable = executable_dir / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from rnagym.config import ConfigFitness\n"
        "Path(os.environ['FITNESS_TEST_ARGUMENTS']).write_text(json.dumps(\n"
        "    {'args': sys.argv[1:], 'cache': os.environ.get('HF_HUB_CACHE'),\n"
        "     'reference': str(ConfigFitness.REFERENCE_FILE),\n"
        "     'checkpoints': str(ConfigFitness.CHECKPOINT_DIR)}))\n"
    )
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        PATH=f"{executable_dir}:{environment['PATH']}",
        SLURM_ARRAY_TASK_ID="24",
        RNAGYM_CHECKPOINT_DIR=str(tmp_path / "shared checkpoints"),
        RNAGYM_DATA_DIR=str(tmp_path / "external data"),
        FITNESS_TEST_ARGUMENTS=str(capture),
    )
    environment.pop("HF_HUB_CACHE", None)
    cache_families = {"aido-rna", "evo", "evo2", "ntv3", "orthrus"}
    for name, family in check_published.MODEL_FAMILIES.items():
        command = f"bash {shlex.quote(str(ConfigFitness.DIR / family.launcher))}"
        subprocess.run(shlex.split(command), env=environment, check=True)
        captured = json.loads(capture.read_text())
        args = captured["args"]
        assert args[args.index("--rows") + 1] == "24"
        assert captured["reference"] == str(
            tmp_path / "external data" / "fitness" / "reference_sheet_final.csv"
        )
        checkpoint_root = tmp_path / "shared checkpoints"
        if name in cache_families:
            assert captured["cache"] == str(checkpoint_root / name / "hub")
            subprocess.run(
                shlex.split(command),
                env=environment | {"HF_HUB_CACHE": str(tmp_path / "custom cache")},
                check=True,
            )
            assert json.loads(capture.read_text())["cache"] == str(
                tmp_path / "custom cache"
            )
        elif name not in ("genslm", "rna-ernie"):
            assert any(value.startswith(str(checkpoint_root) + "/") for value in args)
        assert captured["checkpoints"] == str(checkpoint_root)

    # Archived Evo2 production jobs use these FP8 token budgets
    launcher = ConfigFitness.DIR / check_published.MODEL_FAMILIES["evo2"].launcher
    for model, budget in (
        ("evo2_1b_base", "32768"),
        ("evo2_7b", "16384"),
        ("evo2_20b", "16384"),
        ("evo2_40b", "8192"),
    ):
        command = f"bash {shlex.quote(str(launcher))}"
        subprocess.run(
            shlex.split(command),
            env=environment | {"EVO2_MODEL_NAME": model},
            check=True,
        )
        args = json.loads(capture.read_text())["args"]
        parsed = evo2.parse_args(args[2:])
        assert parsed.model_name == model
        assert evo2.BATCH_TOKENS[parsed.model_name] == int(budget)

    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(REFERENCE_FILE, source / "reference_sheet_final.csv")
    shutil.copytree(ASSAY_DIR, source / "assays")
    shutil.copytree(PREDICTION_DIR.parent, source / "model_predictions")
    monkeypatch.setattr(ConfigFitness, "DATA_DIR", source)
    monkeypatch.setattr(ConfigFitness, "PREDICTION_DIR", source / "model_predictions")
    monkeypatch.setattr(check_published, "CHECK_ROWS", 32)
    monkeypatch.setattr(
        check_published,
        "MODEL_FAMILIES",
        {
            "rna-fm": ModelFamily(
                "baselines/RNA_FM/score_rna_fm.sh", (("rna_fm", None),)
            )
        },
    )
    selected_root = tmp_path / "selection"
    selections = check_published.prepare_fixture(source, selected_root)
    assert (
        selected_root / "reference_sheet_final.csv"
    ).read_bytes() == REFERENCE_FILE.read_bytes()
    assert [selection[0] for selection in selections.values()] == [0, 1, 2]
    assert all(len(rows) == 32 for _, rows in selections.values())

    # Replay frozen predictions to test the launcher and comparison orchestration
    executable.write_text(
        f"#!{sys.executable}\n"
        "import argparse\n"
        "from pathlib import Path\n"
        "import polars as pl\n"
        "from rnagym.config import ConfigFitness\n"
        "from rnagym.fitness.data import parse_row_ids\n"
        "from rnagym.fitness.tasks.check_published import select_rows\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--rows', required=True)\n"
        "parser.add_argument('--output', type=Path)\n"
        "args, _ = parser.parse_known_args()\n"
        f"source = Path({str(source)!r})\n"
        f"with Path({str(capture)!r}).open('a') as log:\n"
        "    log.write(args.rows + '\\n')\n"
        "args.output.mkdir(parents=True, exist_ok=True)\n"
        "for row in parse_row_ids(args.rows):\n"
        "    name = pl.read_csv(ConfigFitness.REFERENCE_FILE)['DMS_ID'][row]\n"
        "    table = pl.read_csv(source / 'model_predictions/rna_fm_4fill' / f'{name}.csv')\n"
        "    table = table.drop_nulls('mutant')\n"
        "    table[select_rows(table, 32)].write_csv(args.output / f'{name}.csv')\n"
    )
    capture.write_text("")
    monkeypatch.setenv("PATH", environment["PATH"])
    results = list(check_published.run_one("rna-fm", time.monotonic() + 30))
    assert [row["row_id"] for row in results] == [0, 1, 2]
    assert all(row["exact"] for row in results)
    assert capture.read_text().splitlines() == ["0,1,2"]
    report_file = tmp_path / "reproduction.json"
    run_cli(check_published.main, f"rna-fm --timeout 30 --report {report_file}")
    report = json.loads(report_file.read_text())
    assert report["status"] == "passed"
    assert report["scope"] == "rna-fm"
    assert report["code_sha256"] == check_published.fingerprint()
    assert len(report["checks"]) == 3
    for record in report["checks"]:
        assert len(record["assay_sha256"]) == len(record["published_sha256"]) == 64
    executable.write_text(
        executable.read_text().replace(
            "name = pl.read_csv(ConfigFitness.REFERENCE_FILE)['DMS_ID'][row]",
            "name = pl.read_csv(ConfigFitness.REFERENCE_FILE)['DMS_ID'][0]",
        )
    )
    with pytest.raises(FileNotFoundError, match="selected assay"):
        run_cli(check_published.main, f"rna-fm --timeout 30 --report {report_file}")
    report = json.loads(report_file.read_text())
    assert report["status"] == "failed"
    assert "selected assay" in report["error"]
    for options in ("--timeout 0", "--timeout nan", "rna-fm --leaderboard-only"):
        with pytest.raises(ValueError):
            run_cli(check_published.main, options)
    with pytest.raises(subprocess.TimeoutExpired):
        program = "__import__('time').sleep(30)"
        check_published.run_command(
            f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}",
            tmp_path,
            time.monotonic() + 0.1,
        )


def test_reproduction_report_failure(tmp_path, monkeypatch):
    """A partial run, missing model family or changed code cannot leave a passing report."""
    import time

    report_file = tmp_path / "reproduction.json"
    report_file.write_text('{"status": "passed"}\n')
    command = f"rna-fm --report {report_file}"

    def reject_native_scoring(command, cwd, deadline):
        assert command == "pixi run --locked -e evmutation check-scoring"
        raise AssertionError("Native scoring failed")

    with monkeypatch.context() as native:
        native.setattr(check_published, "check_leaderboard", lambda *args: None)
        native.setattr(check_published, "validate_family", lambda *args: None)
        native.setattr(check_published, "run_command", reject_native_scoring)
        with pytest.raises(AssertionError, match="Native scoring failed"):
            run_cli(check_published.main, f"--report {report_file}")
        assert json.loads(report_file.read_text())["status"] == "failed"

    def partial_run(environment, deadline):
        assert environment == "rna-fm" and deadline > time.monotonic()
        assert json.loads(report_file.read_text())["status"] == "running"
        yield {"model": "rna_fm", "assay": "completed"}
        raise TimeoutError("fixture deadline")

    monkeypatch.setattr(check_published, "run_one", partial_run)
    with pytest.raises(TimeoutError, match="fixture deadline"):
        run_cli(check_published.main, command)
    report = json.loads(report_file.read_text())
    assert report["status"] == "failed"
    assert report["checks"] == [{"model": "rna_fm", "assay": "completed"}]
    assert report["scope"] == "rna-fm"

    monkeypatch.setattr(check_published, "MODEL_FAMILIES", {})
    with pytest.raises(AssertionError, match="every leaderboard checkpoint"):
        run_cli(check_published.main, f"--report {report_file}")
    assert json.loads(report_file.read_text())["status"] == "failed"

    fingerprints = iter(("before", "after"))
    monkeypatch.setattr(check_published, "fingerprint", lambda: next(fingerprints))
    monkeypatch.setattr(
        check_published, "check_leaderboard", lambda directory, deadline: None
    )
    with pytest.raises(AssertionError, match="changed during reproduction"):
        run_cli(check_published.main, f"--leaderboard-only --report {report_file}")
    report = json.loads(report_file.read_text())
    assert report["status"] == "failed"
    assert report["scope"] == "leaderboard"

    def broken_runtime():
        raise ImportError("fixture CUDA library")

    report_file.write_text('{"status": "passed"}\n')
    monkeypatch.setattr(check_published, "fingerprint", lambda: "before")
    monkeypatch.setattr(check_published, "runtime_versions", broken_runtime)
    with pytest.raises(ImportError, match="fixture CUDA library"):
        run_cli(check_published.main, f"--leaderboard-only --report {report_file}")
    report = json.loads(report_file.read_text())
    assert report["status"] == "failed"
    assert report["checks"] == []


def test_genslm_causal_likelihood(tmp_path, monkeypatch):
    """Check causal targets, score direction, padding and validation independently."""
    from scipy.special import logsumexp

    from rnagym.fitness.baselines.GenSLM import compute_fitness as genslm

    class TokenDataset(torch.utils.data.Dataset):
        def __init__(self, sequences, length, tokenizer):
            self.sequences = sequences
            self.length = length

        def __getitem__(self, index):
            sequence = self.sequences[index]
            ids = [
                sum("ACGT".index(base) for base in sequence[start : start + 3])
                for start in range(0, len(sequence), 3)
            ]
            return {
                "input_ids": torch.tensor(ids + [0] * (self.length - len(ids))),
                "attention_mask": torch.tensor(
                    [1] * len(ids) + [0] * (self.length - len(ids))
                ),
            }

        def __len__(self):
            return len(self.sequences)

    class TokenModel(torch.nn.Module):
        seq_length = 40
        tokenizer = SimpleNamespace(num_special_tokens_to_add=lambda: 0)

        def forward(self, input_ids, attention_mask, output_hidden_states):
            assert not output_hidden_states
            logits = torch.sin(input_ids[..., None] + torch.arange(10).float())
            return SimpleNamespace(logits=logits)

    monkeypatch.setitem(
        sys.modules, "genslm", SimpleNamespace(SequenceDataset=TokenDataset)
    )
    assay_dir = tmp_path / "assays"
    assay_dir.mkdir()
    name = "Tome_2014_GFP_aptamer"
    assay_file = assay_dir / f"{name}.csv"
    assay = pl.read_csv(ASSAY_DIR / assay_file.name)
    assay.write_csv(assay_file)
    monkeypatch.setattr(ConfigFitness, "ASSAY_DIR", assay_dir)
    monkeypatch.setattr(ConfigFitness, "REFERENCE_FILE", REFERENCE_FILE)
    args = genslm.parse_args(shlex.split(f"--rows 2 --output {tmp_path / 'scores'}"))
    shutil.copy2(ASSAY_DIR / ASSAY_NAMES[1], assay_dir / ASSAY_NAMES[1])
    loads = []

    def load_model(*args, **kwargs):
        loads.append(TokenModel())
        return loads[-1]

    args.rows = "1-2"
    genslm.main(args, load_model)
    assert len(loads) == 1
    assert pl.read_csv(args.output / ASSAY_NAMES[1]).height == 4175
    args.rows = "2"
    result_file = args.output / assay_file.name
    result = pl.read_csv(result_file)
    expected = []
    for sequence in assay["sequence"]:
        ids = np.array(
            [
                sum("ACGU".index(base) for base in sequence[start : start + 3])
                for start in range(0, len(sequence), 3)
            ]
        )
        logits = np.sin(ids[:, None] + np.arange(10))
        expected.append(
            np.mean(
                logits[np.arange(len(ids) - 1), ids[1:]]
                - logsumexp(logits[:-1], axis=1)
            )
        )
    np.testing.assert_allclose(result["logit_scores"], expected, rtol=3e-7)
    assert_frame_equal(result.select(assay.columns), assay)
    assert not result["mutated_sequence"].str.contains("U").any()

    # Real assay sequences with different lengths exercise padding in the same batch
    sequences = [
        pl.read_csv(ASSAY_DIR / name)["sequence"][0].replace("U", "T")
        for name in ASSAY_NAMES
    ]
    TokenModel.seq_length = 512
    individual = np.concatenate(
        [
            genslm.sequence_log_likelihoods(TokenModel(), [sequence], "cpu")
            for sequence in sequences
        ]
    )
    batched = genslm.sequence_log_likelihoods(TokenModel(), sequences, "cpu")
    np.testing.assert_allclose(batched, individual, rtol=2e-7)
    for short in (["ACG"], [sequences[0], "ACG"]):
        with pytest.raises(ValueError, match="at least two tokens"):
            genslm.sequence_log_likelihoods(TokenModel(), short, "cpu")
    with pytest.raises(ValueError, match="only A, C, G and T"):
        genslm.sequence_log_likelihoods(TokenModel(), ["ACGTNN"], "cpu")
    original = result_file.read_bytes()
    TokenModel.seq_length = 2
    with pytest.raises(ValueError, match="input length limit"):
        genslm.main(args, lambda *args, **kwargs: TokenModel())
    assert result_file.read_bytes() == original
    assay.with_columns(pl.lit(None, dtype=pl.String).alias("sequence")).write_csv(
        assay_file
    )
    with pytest.raises(ValueError, match="missing or empty sequences"):
        genslm.main(args, lambda *args, **kwargs: TokenModel())
    assert result_file.read_bytes() == original


def test_reproduction_timeout_stops_inference(tmp_path):
    """A parent deadline stops a scoring process in a nested process group."""
    import os
    import subprocess
    import time

    pid_file = tmp_path / "inference.pid"
    child = tmp_path / "inference.py"
    child.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    worker = tmp_path / "family.py"
    report = tmp_path / "family.json"
    inference_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(child))}"
    worker.write_text(
        "from pathlib import Path\n"
        "from rnagym.fitness.tasks import check_published as check\n"
        "def run_one(environment, deadline):\n"
        "    yield {'model': 'started'}\n"
        f"    check.run_command({inference_command!r}, Path.cwd(), deadline)\n"
        "check.run_one = run_one\n"
        f"check.main(['rna-fm', '--report', {str(report)!r}])\n"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        check_published.run_command(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(worker))}",
            tmp_path,
            started + 4,
        )
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert json.loads(report.read_text())["status"] == "failed"
