"""CPU checks of the clinical benchmark: scorers against hand-computable fake
models, staging, shards and merge, evaluation, and the leaderboard on a fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from rnagym.clinical import common
from rnagym.clinical.models import MODELS
from rnagym.clinical.score_evo2 import (
    reverse_complement,
    score_rows,
    window_log_likelihoods,
)
from rnagym.clinical.score_masked import build_buckets, finalize, run_buckets
from rnagym.clinical.tasks import evaluate, leaderboard
from rnagym.clinical.tasks.merge import merge
from rnagym.clinical.tasks.prepare import check_row, delete_placeholder
from rnagym.clinical.windows import crop_bounds
from rnagym.fitness.baselines.masked_lm.adapter import MaskedLMAdapter
from rnagym.fitness.baselines.masked_lm.engine import window_contexts
from rnagym.fitness.baselines.masked_lm.strategies import MASK_CHAR, TaskTable

VOCAB = 12
PREFIX, SUFFIX, MASK, PAD, UNK = 1, 9, 2, 0, 3
BASE_IDS = {"A": 4, "C": 5, "G": 6, "T": 7}


def naive_crop(length: int, lo: int, hi: int, budget: int) -> int:
    """Independent statement of the crop rule: centre, then clamp into range."""
    if length <= budget:
        return 0
    return min(max((lo + hi) // 2 - budget // 2, 0), length - budget)


# ---------------------------------------------------------------- crops


def test_crop_matches_window_contexts():
    rng = random.Random(0)
    for _ in range(500):
        length = rng.randint(1, 60)
        budget = rng.randint(1, 70)
        pos = rng.randrange(length)
        seq = "".join(rng.choice("ACGT") for _ in range(length))
        context = seq[:pos] + MASK_CHAR + seq[pos + 1 :]
        table = TaskTable(
            contexts=[context],
            ctx_id=np.array([0], dtype=np.int32),
            pos=np.array([pos], dtype=np.int32),
            base=np.array([0], dtype=np.int8),
            row=np.array([0], dtype=np.int32),
            strategy=np.array([0], dtype=np.int8),
            sign=np.array([1], dtype=np.int8),
            scorable=np.array([True]),
            strategies=("wt_fill",),
        )
        window_contexts(table, budget)
        start, end = crop_bounds(length, pos, pos, budget)
        assert table.contexts[0] == context[start:end]
        assert int(table.pos[0]) == pos - start
        assert start == naive_crop(length, pos, pos, budget)


def test_crop_pair_insertion_at_end_and_containment():
    crop = common.crop_pair("ACGTACGT", "ACGTACGTGG", 8, "", "GG", 100)
    assert crop.contained and crop.ref == "ACGTACGT" and crop.alt == "ACGTACGTGG"
    # A 4-base deletion with a 3-base cap cannot be contained, and must not crash
    crop = common.crop_pair("AACCCCAA", "AAAAGGGG", 2, "CCCC", "", 3)
    assert not crop.contained and len(crop.ref) == 3
    crop = common.crop_pair("A" * 20, "A" * 10 + "G" * 5 + "A" * 10, 10, "", "GGGGG", 4)
    assert not crop.contained  # the inserted 5 bases cannot fit a 4-base window


# ---------------------------------------------------------------- masked scorer


class FakeAdapter(MaskedLMAdapter):
    """Logits depend on the position and both neighbouring tokens."""

    name = "fake"
    bases = "ACGT"
    n_special_tokens = 2
    default_batch_size = 3
    default_max_batch_tokens = 10_000

    def __init__(self, multiple: int = 1):
        self.multiple = multiple
        if multiple > 1:
            self.context_pad_char = "N"

    def context_length_for(self, length: int) -> int:
        return -(-length // self.multiple) * self.multiple

    def load(self, args=None):
        self.base_ids = dict(BASE_IDS)
        self.mask_id, self.pad_id, self.unk_id = MASK, PAD, UNK
        self.prefix_ids, self.suffix_ids = [PREFIX], [SUFFIX]
        self.device = "cpu"

    def logits_at(self, input_ids, attention_mask, rows, cols):
        left = input_ids[rows, cols - 1].double()
        right = input_ids[rows, cols + 1].double()
        v = torch.arange(VOCAB, dtype=torch.double)
        return (
            torch.cos(0.37 * cols.double()[:, None] * (v + 1))
            + 0.11 * left[:, None] * torch.sin(v)
            + 0.07 * right[:, None] * (v + 1) / VOCAB
        )


def naive_logprob(window: str, p: int, base: str, adapter: FakeAdapter) -> float:
    """Masked log p(base) at window position p, computed without the engine."""
    padded = window
    if adapter.multiple > 1:
        padded += "N" * (adapter.context_length_for(len(window)) - len(window))
    tokens = (
        [PREFIX]
        + [MASK if i == p else BASE_IDS.get(c, UNK) for i, c in enumerate(padded)]
        + [SUFFIX]
    )
    col = p + 1
    logits = [
        math.cos(0.37 * col * (v + 1))
        + 0.11 * tokens[col - 1] * math.sin(v)
        + 0.07 * tokens[col + 1] * (v + 1) / VOCAB
        for v in range(VOCAB)
    ]
    norm = math.log(sum(math.exp(x) for x in logits))
    return logits[BASE_IDS[base]] - norm


def naive_masked(row: tuple, cap: int, k: int, adapter: FakeAdapter) -> float:
    cohort, vi, ref, alt, seq_ref, seq_alt = row
    lo = min(vi, len(seq_ref) - 1)
    hi = max(lo, vi + len(ref) - 1)
    start = naive_crop(len(seq_ref), lo, hi, cap)
    wref = seq_ref[start : start + cap]
    walt = seq_alt[start : start + cap]
    site = vi - start
    if cohort == "snv":
        return naive_logprob(wref, site, alt, adapter) - naive_logprob(
            wref, site, ref, adapter
        )
    means = []
    for window, allele in ((walt, alt), (wref, ref)):
        positions = range(max(0, site - k), min(len(window), site + len(allele) + k))
        means.append(
            sum(naive_logprob(window, p, window[p], adapter) for p in positions)
            / len(positions)
        )
    return means[0] - means[1]


def masked_rows(rng: random.Random) -> list[tuple]:
    rows = []
    base = "".join(rng.choice("ACGT") for _ in range(40))
    for vi in (0, 1, 7, 20, 38, 39):  # first, second, middle and last positions
        for alt in "ACGT":
            if alt != base[vi]:
                rows.append(
                    ("snv", vi, base[vi], alt, base, base[:vi] + alt + base[vi + 1 :])
                )
    short = "".join(rng.choice("ACGT") for _ in range(9))  # shorter than the cap
    rows.append(
        (
            "snv",
            4,
            short[4],
            "A" if short[4] != "A" else "C",
            short,
            short[:4] + ("A" if short[4] != "A" else "C") + short[5:],
        )
    )
    # Deletion and insertion, equal-length windows (pre-mRNA style) and unequal (mRNA style)
    rows.append(("nonsnv", 15, base[15:18], "", base, base[:15] + base[18:] + "GGG"))
    rows.append(("nonsnv", 15, "", "TT", base, base[:15] + "TT" + base[15:38]))
    rows.append(("nonsnv", 30, base[30:32], "", base, base[:30] + base[32:]))
    rows.append(
        ("nonsnv", 40, "", "GA", base, base + "GA")
    )  # insertion after the last base
    return rows


@pytest.mark.parametrize("multiple", [1, 8])
@pytest.mark.parametrize("cap", [12, 64])
def test_masked_scores_match_naive(cap, multiple):
    adapter = FakeAdapter(multiple)
    adapter.load()
    rows = masked_rows(random.Random(1))
    k = 3
    buckets, meta = build_buckets(
        rows, adapter.bases, adapter.canonicalize_sequence, cap, k
    )
    total = run_buckets(adapter, buckets, len(rows), adapter.default_max_batch_tokens)
    result = finalize(meta, [r[0] for r in rows], total)
    for r, row in enumerate(rows):
        if not result["contained"][r]:
            assert row[0] != "snv" and math.isnan(result["score"][r])
            continue
        expected = naive_masked(row, cap, k, adapter)
        # The engine takes log-softmax in float32
        assert result["score"][r] == pytest.approx(expected, abs=1e-5), (r, row[:4])
        if row[0] == "snv":
            assert result["site_llr"][r] == result["score"][r]


def test_masked_shared_contexts_are_deduplicated():
    adapter = FakeAdapter()
    adapter.load()
    seq = "ACGTACGTACGT"
    rows = [("snv", 5, "C", alt, seq, seq[:5] + alt + seq[6:]) for alt in "AGT"]
    buckets, _ = build_buckets(
        rows, adapter.bases, adapter.canonicalize_sequence, 100, 3
    )
    assert sum(len(b.contexts) for b in buckets.values()) == 1


# ---------------------------------------------------------------- Evo 2 scorer


def fake_forward(ids: torch.Tensor) -> torch.Tensor:
    """Logits that depend on the current and previous token only."""
    prev = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1).double()
    v = torch.arange(256, dtype=torch.double)
    return torch.sin(
        0.013 * ids.double()[..., None] * (v + 1) + 0.007 * prev[..., None]
    )


def naive_ll(seq: str) -> float:
    ids = [ord(c) for c in seq]
    total = 0.0
    for t in range(1, len(ids)):
        cur, prev = ids[t - 1], ids[t - 2] if t >= 2 else 0
        logits = [math.sin(0.013 * cur * (v + 1) + 0.007 * prev) for v in range(256)]
        norm = math.log(sum(math.exp(x) for x in logits))
        total += logits[ids[t]] - norm
    return total


def test_window_log_likelihood_is_batch_invariant_and_exact():
    rng = random.Random(2)
    seqs = ["".join(rng.choice("ACGT") for _ in range(n)) for n in (5, 5, 9, 5, 9, 2)]
    expected = np.array([naive_ll(s) for s in seqs])
    for budget in (1, 10, 1000):
        got = window_log_likelihoods(fake_forward, seqs, budget, "cpu")
        assert np.allclose(got, expected, rtol=1e-5, atol=1e-4)  # float32 log-softmax


def test_evo_rows_match_naive_deltas():
    rng = random.Random(3)
    base = "".join(rng.choice("ACGT") for _ in range(30))
    rows = [
        (
            12,
            base[12],
            "A" if base[12] != "A" else "G",
            base,
            base[:12] + ("A" if base[12] != "A" else "G") + base[13:],
        ),
        (
            0,
            base[0],
            "C" if base[0] != "C" else "T",
            base,
            ("C" if base[0] != "C" else "T") + base[1:],
        ),
        (10, base[10:13], "", base, base[:10] + base[13:] + "AAA"),
        (10, "", "GG", base, base[:10] + "GG" + base[10:]),
    ]
    for cap in (11, 64):
        result, _ = score_rows(fake_forward, rows, cap, 50, "cpu", with_rc=True)
        for r, (vi, ref, alt, seq_ref, seq_alt) in enumerate(rows):
            lo = vi
            hi = max(lo, vi + len(ref) - 1)
            start = naive_crop(len(seq_ref), lo, hi, cap)
            wr, wa = seq_ref[start : start + cap], seq_alt[start : start + cap]
            fwd = naive_ll(wa) / (len(wa) - 1) - naive_ll(wr) / (len(wr) - 1)
            rc = naive_ll(reverse_complement(wa)) / (len(wa) - 1) - naive_ll(
                reverse_complement(wr)
            ) / (len(wr) - 1)
            if not result["contained"][r]:
                assert math.isnan(result["score"][r])
                continue
            assert result["delta_fwd"][r] == pytest.approx(fwd, abs=1e-5)
            assert result["delta_rc"][r] == pytest.approx(rc, abs=1e-5)
            assert (
                result["score"][r] == result["delta_fwd"][r]
            )  # the paper's forward-only score


def test_evo_skips_uncontained_rows_before_inference():
    # Deletion CGT of ACGT with cap 2: uncontained, and its 1-base alternate must not reach the model
    calls = []

    def forward(ids):
        calls.append(tuple(ids.shape))
        return fake_forward(ids)

    rows = [(1, "CGT", "", "ACGT", "A"), (1, "C", "G", "ACGT", "AGGT")]
    result, n = score_rows(forward, rows, 2, 50, "cpu")
    assert not result["contained"][0] and math.isnan(result["score"][0])
    assert result["contained"][1] and math.isfinite(result["score"][1])
    assert all(shape[1] >= 2 for shape in calls)
    assert math.isnan(result["delta_rc"][1])  # reverse complement only on request


def test_reverse_complement():
    assert reverse_complement("AACGT") == "ACGTT"


# ---------------------------------------------------------------- staging


def test_placeholder_deletion_shifts_downstream_index():
    seq_ref = "ACNGTACG"
    seq_alt = "ACNGTTCG"
    out = delete_placeholder(seq_ref, seq_alt, 5, "A", "T")
    assert out == ("ACGTACG", "ACGTTCG", 4, True)
    out = delete_placeholder("ACGTANCG", "ACTTANCG", 2, "G", "T")
    assert out == ("ACGTACG", "ACTTACG", 2, True)
    # Deletion upstream of a downstream placeholder: the N moves by the indel length
    out = delete_placeholder("ACGTTANCG", "ACTANCGGG", 2, "GT", "")
    assert out[2] == 2 and out[3]
    with pytest.raises(ValueError):
        delete_placeholder("ACNGT", "ACTGT", 2, "N", "T")


def test_check_row_rejects_bad_rows():
    check_row("ACGT", "ACTT", 2, "G", "T", True, True)
    with pytest.raises(ValueError):
        check_row("ACGT", "TCTT", 2, "G", "T", True, True)  # second difference
    with pytest.raises(ValueError):
        check_row("ACGT", "ACTT", 1, "G", "T", True, True)  # wrong index
    # Indels: fixed-length window and whole-unit view
    check_row("AACCGGTT", "AAGGTTAC", 2, "CC", "", False, True)
    check_row("AACCGGTT", "AAGGTT", 2, "CC", "", False, False)
    with pytest.raises(ValueError):
        check_row(
            "AACCGG", "AATTAT", 2, "CC", "TT", False, True
        )  # extra change downstream
    with pytest.raises(ValueError):
        check_row(
            "AACCGGTT", "AAGGTT", 2, "CC", "", False, True
        )  # fixed window, unequal lengths
    with pytest.raises(ValueError):
        check_row(
            "AACCGGTT", "AAGGTTAC", 2, "CC", "", False, False
        )  # whole unit, wrong length


# ---------------------------------------------------------------- shards and merge


def staged_fixture(tmp_path, n=40, name="toy"):
    groups = [i // 3 for i in range(n)]
    frame = pl.DataFrame({"row_id": list(range(n)), "group_id": groups})
    frame.write_parquet(tmp_path / f"{name}.parquet")
    digest = hashlib.sha256((tmp_path / f"{name}.parquet").read_bytes()).hexdigest()
    (tmp_path / f"{name}.summary.json").write_text(
        json.dumps({"staged_sha256": digest})
    )
    return np.array(groups)


def write_toy_shards(
    staged, out, groups, num_shards, name="toy", skip=None, mixed=False, row_ids=None
):
    shard_of = common.shard_of_groups(groups, num_shards)
    for k in range(num_shards):
        if k == skip:
            continue
        ids = np.flatnonzero(shard_of == k)
        if row_ids is not None:
            ids = ids[np.isin(ids, row_ids)]
        frame = pl.DataFrame(
            {"row_id": ids, "contained": [True] * len(ids), "score": [0.5] * len(ids)}
        )
        spec = {
            "name": name,
            "staged_sha256": common.staged_sha256(staged, name),
            "row_ids_file": None,
            "num_shards": num_shards,
            "model": "other" if mixed and k == 0 else "toy",
        }
        manifest = {"spec": spec, "spec_hash": common.spec_hash(spec), "shard": k}
        common.write_shard(out, k, num_shards, frame, manifest)
    return out


def test_shards_keep_groups_together_and_cover_all_rows():
    groups = np.repeat(
        np.arange(17), [1, 5, 2, 9, 1, 1, 3, 4, 2, 8, 1, 1, 1, 6, 2, 2, 3]
    )
    for n in (1, 2, 5, 17, 40):
        shard = common.shard_of_groups(groups, n)
        for g in np.unique(groups):
            assert len(set(shard[groups == g])) == 1
        assert shard.min() >= 0 and shard.max() < n


def test_merge_accepts_complete_and_rejects_broken(tmp_path):
    groups = staged_fixture(tmp_path)
    out = write_toy_shards(tmp_path, tmp_path / "ok", groups, 4)
    merged = merge(out, tmp_path, "toy", None)
    assert merged["row_id"].to_list() == list(range(40))

    missing = write_toy_shards(tmp_path, tmp_path / "missing", groups, 4, skip=2)
    with pytest.raises(ValueError):
        merge(missing, tmp_path, "toy", None)

    mixed = write_toy_shards(tmp_path, tmp_path / "mixed", groups, 4, mixed=True)
    with pytest.raises(ValueError):
        merge(mixed, tmp_path, "toy", None)

    # Scores of another view with the same row ids must not pass
    staged_fixture(tmp_path, name="other")
    other = write_toy_shards(tmp_path, tmp_path / "other", groups, 4, name="other")
    with pytest.raises(ValueError):
        merge(other, tmp_path, "toy", None)

    # A manifest whose spec was edited no longer hashes to its spec_hash
    edited = out / "shard_0002_of_0004.json"
    manifest = json.loads(edited.read_text())
    manifest["spec"]["model"] = "edited"
    edited.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        merge(out, tmp_path, "toy", None)

    tampered = tmp_path / "ok2"
    write_toy_shards(tmp_path, tampered, groups, 4)
    pl.DataFrame({"row_id": [999], "contained": [True], "score": [0.1]}).write_parquet(
        tampered / "shard_0001_of_0004.parquet"
    )
    with pytest.raises(ValueError):
        merge(tampered, tmp_path, "toy", None)


def test_merge_rejects_stale_staging(tmp_path):
    groups = staged_fixture(tmp_path)
    out = write_toy_shards(tmp_path, tmp_path / "ok", groups, 2)
    staged_fixture(tmp_path, n=41)  # restaged: different file, new summary
    with pytest.raises(ValueError):
        merge(out, tmp_path, "toy", None)


def test_requested_rows_must_exist(tmp_path):
    staged_fixture(tmp_path)
    good = tmp_path / "good.txt"
    good.write_text("1\n5\n7\n")
    assert common.expected_row_ids(tmp_path, "toy", good).tolist() == [1, 5, 7]
    bad = tmp_path / "bad.txt"
    bad.write_text("1\n999\n")
    with pytest.raises(ValueError):
        common.expected_row_ids(tmp_path, "toy", bad)
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError):
        common.expected_row_ids(tmp_path, "toy", empty)


def test_write_shard_claims_exclusively(tmp_path):
    frame = pl.DataFrame({"row_id": [0], "contained": [True], "score": [0.0]})
    common.write_shard(
        tmp_path, 0, 1, frame, {"spec": {"num_shards": 1}, "spec_hash": "a", "shard": 0}
    )
    with pytest.raises(FileExistsError):
        common.write_shard(
            tmp_path,
            0,
            1,
            frame,
            {"spec": {"num_shards": 1}, "spec_hash": "a", "shard": 0},
        )
    # A competing writer that claimed first blocks a second one even before any output exists
    (tmp_path / "shard_0000_of_0002.claim").write_text("other host\n")
    with pytest.raises(FileExistsError):
        common.write_shard(
            tmp_path,
            0,
            2,
            frame,
            {"spec": {"num_shards": 2}, "spec_hash": "a", "shard": 0},
        )


# ---------------------------------------------------------------- evaluation


def test_auroc_matches_sklearn_with_ties_and_weights():
    from sklearn.metrics import roc_auc_score

    from rnagym.clinical.tasks.evaluate import auroc, auroc_weighted

    rng = np.random.default_rng(0)
    score = rng.integers(0, 6, 300).astype(float)  # many ties
    label = (rng.random(300) < 0.3).astype(float)
    assert auroc(score, label) == pytest.approx(roc_auc_score(label, score))
    # Integer weights equal row repetition
    weight = rng.integers(0, 3, 300).astype(float)
    uniq, inverse = np.unique(score, return_inverse=True)
    repeated = np.repeat(np.arange(300), weight.astype(int))
    expected = roc_auc_score(label[repeated], score[repeated])
    assert auroc_weighted(inverse, len(uniq), label, weight) == pytest.approx(expected)
    assert math.isnan(auroc(score, np.zeros(300)))


def test_consequence_baseline_is_out_of_fold():
    from rnagym.clinical.tasks.evaluate import consequence_baseline

    genes = [f"g{i}" for i in range(200)]
    meta = pl.DataFrame(
        {
            "gene": genes,
            "binary_label": [1 if i % 2 else 0 for i in range(200)],
            "consequence_tx": [
                "nonsense" if i % 2 else "synonymous_variant" for i in range(200)
            ],
        }
    )
    out = consequence_baseline(meta)
    assert np.all(out[1::2] > 0.9) and np.all(out[::2] < 0.1)
    # A gene's own label never enters its prediction: a lone category gets the fold rate
    meta = meta.with_columns(
        pl.Series(
            "consequence_tx", ["only_here"] + meta["consequence_tx"].to_list()[1:]
        )
    )
    assert 0.3 < consequence_baseline(meta)[0] < 0.7


def test_write_shard_refuses_unclaimed_existing_outputs(tmp_path):
    frame = pl.DataFrame({"row_id": [0], "contained": [True], "score": [0.0]})
    manifest = {"spec": {"num_shards": 1}, "spec_hash": "a", "shard": 0}
    for leftover in ("shard_0000_of_0001.parquet", "shard_0000_of_0001.json"):
        out = tmp_path / leftover.split(".")[1]
        out.mkdir()
        (out / leftover).write_text("old")
        with pytest.raises(FileExistsError):
            common.write_shard(out, 0, 1, frame, manifest)
        assert (out / leftover).read_text() == "old"


def test_group_columns_survive_leading_one_class_groups():
    from rnagym.clinical.tasks.evaluate import METRICS, group_record, records_frame

    # mt_trna_cca: the first records are non-SNV groups with no benign rows, so no AUROC
    weights = np.ones((1, 4))
    one_class = group_record(
        "f",
        "nonsnv",
        "model_rows",
        "k",
        "g",
        "m",
        False,
        np.zeros(4),
        np.ones(4),
        np.ones(4, dtype=bool),
        weights,
        0,
    )
    assert all(k in one_class and one_class[k] is None for k in METRICS)
    both = group_record(
        "f",
        "snv",
        "model_rows",
        "k",
        "g",
        "m",
        False,
        np.arange(4.0),
        np.array([0, 0, 1, 1.0]),
        np.ones(4, dtype=bool),
        weights,
        0,
    )
    table = records_frame([one_class] * 150 + [both])
    assert table["auroc"].to_list()[-1] == 1.0 and table["auroc"].null_count() == 150


def test_mito_clusters_are_the_processed_rna_units():
    from rnagym.clinical.tasks.evaluate import MITO_UNITS, mito_clusters

    assert len(MITO_UNITS) == 35 and "MT-5" not in MITO_UNITS
    assert "MT-ATP8/ATP6" in MITO_UNITS and "MT-ATP8" not in MITO_UNITS
    meta = pl.DataFrame({"unit": ["MT-TF", "MT-ND4L/ND4", "MT-TF"]})
    assert mito_clusters(meta).tolist() == ["MT-TF", "MT-ND4L/ND4", "MT-TF"]
    # A row outside the processed RNAs has no cluster in the RNA view
    for unit in ("", None, "MT-5"):
        with pytest.raises(ValueError):
            mito_clusters(pl.DataFrame({"unit": ["MT-TF", unit]}))


def test_mito_groups_have_one_benign_set():
    from rnagym.clinical.tasks.evaluate import groups_for

    meta = pl.DataFrame(
        {
            "binary_label": [1, 0, 0, 1],
            "mt_stratum": ["tRNA", "tRNA", "missense_variant", "missense_variant"],
        }
    )
    groups = groups_for(meta, "mt_rna")
    # Mito files have no separate "all" group type: the all-rows group is a stratum of the one set
    assert {kind for kind, _, _ in groups} == {"mito:all_benign"}
    masks = {
        group: mask.tolist()
        for kind, group, mask in groups
        if kind == "mito:all_benign"
    }
    assert masks == {
        "all": [True] * 4,
        "tRNA": [True, True, False, False],
        "missense_variant": [False, False, True, True],
    }


def test_settings_are_written_per_file_with_the_evaluator_hash(tmp_path):
    args = argparse.Namespace(names=["a", "b"], bootstrap=1000, seed=1)
    evaluate.write_settings(tmp_path, "a", args)
    settings = json.loads((tmp_path / "a.settings.json").read_text())
    assert settings["names"] == ["a"] and settings["bootstrap"] == 1000
    assert (
        settings["evaluator_sha256"]
        == hashlib.sha256(open(evaluate.__file__, "rb").read()).hexdigest()
    )
    assert not (tmp_path / "settings.json").exists()


# ---------------------------------------------------------------- checkpoint hashes


def test_checkpoint_hashes_on_a_cache_miss(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "a.bin").write_bytes(b"first")
    (weights / "b.json").write_text("{}")
    cache = tmp_path / "cache" / "sha256.json"
    identity = common.checkpoint_identity(weights, None, cache)
    digests = {Path(r["path"]).name: r["sha256"] for r in identity["local"]}
    assert digests == {
        "a.bin": hashlib.sha256(b"first").hexdigest(),
        "b.json": hashlib.sha256(b"{}").hexdigest(),
    }
    assert len(json.loads(cache.read_text())) == 2
    # Moved weights are hashed again under their new identity, to the same digest
    moved = tmp_path / "moved"
    shutil.copytree(weights, moved)
    identity = common.checkpoint_identity(moved / "a.bin", None, cache)
    assert identity["local"][0]["sha256"] == digests["a.bin"]
    assert len(json.loads(cache.read_text())) == 3


def test_checkpoint_hash_follows_a_same_size_replacement(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"aaaa")
    cache = tmp_path / "sha256.json"
    before = common.checkpoint_identity(path, None, cache)["local"][0]
    stat = path.stat()
    path.write_bytes(b"bbbb")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # size and mtime as before
    after = common.checkpoint_identity(path, None, cache)["local"][0]
    assert (after["size"], after["mtime_ns"]) == (before["size"], before["mtime_ns"])
    assert after["sha256"] == hashlib.sha256(b"bbbb").hexdigest()


def test_hub_checkpoints_need_a_pinned_revision(tmp_path):
    cache = tmp_path / "sha256.json"
    assert common.checkpoint_identity("org/model", "abc", cache) == {
        "hub_id": "org/model",
        "revision": "abc",
    }
    with pytest.raises(ValueError):
        common.checkpoint_identity("org/model", None, cache)


# ---------------------------------------------------------------- workflow on the fixture

FIXTURE = Path(__file__).parent / "fixtures" / "clinical"
FILES = (
    "premrna_10101_snv",
    "mature_mrna_snv",
    "premrna_10101_indel",
    "mature_mrna_indel",
    "mt_rna",
    "mt_trna_cca",
)
# The fixture's expected tables were made with 20 draws (fixtures/clinical/README.md)
DRAWS, SEED = 20, 20261008


def fixture_scores(tmp_path: Path) -> Path:
    """Lay the fixture's scores out as merged/<file>/<model>.parquet."""
    merged = tmp_path / "merged"
    for name in FILES:
        wide = pl.read_parquet(FIXTURE / "scores" / f"{name}.parquet")
        (merged / name).mkdir(parents=True)
        for model in MODELS:
            wide.select("row_id", pl.col(model).alias("score")).write_parquet(
                merged / name / f"{model}.parquet"
            )
    return merged


def read_table(path: Path) -> pl.DataFrame:
    """Read a CSV the way the comparisons need it: every column, no inference gaps."""
    return pl.read_csv(path, infer_schema_length=None)


def test_evaluation_and_leaderboard_reproduce_the_fixture(tmp_path):
    merged = fixture_scores(tmp_path)
    out = tmp_path / "evaluation"
    evaluate.main(
        [
            "--staged",
            str(FIXTURE / "staged"),
            "--merged",
            str(merged),
            "--names",
            *FILES,
            "--out",
            str(out),
            "--bootstrap",
            str(DRAWS),
            "--seed",
            str(SEED),
        ]
    )
    expected = FIXTURE / "expected"
    for path in sorted(expected.glob("*_*.*.csv")):
        assert_frame_equal(
            read_table(out / path.name),
            read_table(path),
            rel_tol=0,
            abs_tol=1e-12,
        )
    board = tmp_path / "leaderboard"
    leaderboard.main(["--input", str(out), "--output", str(board)])
    for name in ("leaderboard", "units", "classes", "mito_strata"):
        assert_frame_equal(
            read_table(board / f"{name}.csv"),
            read_table(expected / f"{name}.csv"),
            rel_tol=0,
            abs_tol=1e-12,
        )
    readme = (board / "README.md").read_text()
    assert readme == (FIXTURE / "leaderboard.md").read_text()


def test_leaderboard_rejects_a_missing_model(tmp_path):
    merged = fixture_scores(tmp_path)
    out = tmp_path / "evaluation"
    evaluate.main(
        ["--staged", str(FIXTURE / "staged"), "--merged", str(merged)]
        + ["--names", *FILES, "--out", str(out), "--bootstrap", "0"]
        + ["--models", *MODELS[:-1]]
    )
    with pytest.raises(ValueError):
        leaderboard.write_tables(out, tmp_path / "board", MODELS)
    # A model without scores cannot be evaluated either
    (merged / "mt_rna" / f"{MODELS[0]}.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate.evaluate("mt_rna", FIXTURE / "staged", merged, MODELS, 0, SEED)


def test_leaderboard_scores_are_plain_means(tmp_path):
    classes = read_table(FIXTURE / "expected" / "classes.csv")
    board = read_table(FIXTURE / "expected" / "leaderboard.csv")
    units = read_table(FIXTURE / "expected" / "units.csv")
    mito = (
        classes.filter(pl.col("unit") == "mito")
        .group_by("model")
        .agg(pl.col("auroc").mean())
    )
    joined = board.join(mito, on="model")
    assert (joined["mito"] - joined["auroc"]).abs().max() < 1e-11
    nuclear = (
        units.filter(pl.col("unit") != "mito")
        .group_by("model")
        .agg(pl.col("macro").mean())
    )
    joined = board.join(nuclear, on="model")
    assert (joined["clinvar"] - joined["macro"]).abs().max() < 1e-11


def test_merge_rejects_shards_of_another_model(tmp_path):
    from rnagym.clinical.tasks import merge as merge_task

    groups = staged_fixture(tmp_path)
    # Shards of model "toy" filed under another model's directory
    misfiled = write_toy_shards(
        tmp_path, tmp_path / "scores" / "toy" / "rinalmo", groups, 2
    )
    with pytest.raises(ValueError):
        merge(misfiled, tmp_path, "toy", None, "rinalmo")
    assert merge(misfiled, tmp_path, "toy", None)["row_id"].len() == 40
    with pytest.raises(ValueError):
        merge_task.main(
            ["--score-dir", str(tmp_path / "scores"), "--staged", str(tmp_path)]
            + ["--merged", str(tmp_path / "merged")]
        )
    assert not (tmp_path / "merged" / "toy" / "rinalmo.parquet").exists()


def test_merge_writes_every_model_and_rejects_an_empty_score_dir(tmp_path):
    from rnagym.clinical.tasks import merge as merge_task

    groups = staged_fixture(tmp_path)
    write_toy_shards(tmp_path, tmp_path / "scores" / "toy" / "toy", groups, 3)
    merged = tmp_path / "merged"
    args = ["--staged", str(tmp_path), "--merged", str(merged)]
    merge_task.main(["--score-dir", str(tmp_path / "scores"), *args])
    assert pl.read_parquet(merged / "toy" / "toy.parquet")["row_id"].len() == 40
    with pytest.raises(FileNotFoundError):
        merge_task.main(["--score-dir", str(tmp_path / "missing"), *args])


def test_evo2_40b_check_is_reused_only_for_the_file_that_passed(tmp_path, monkeypatch):
    from rnagym.clinical import score_evo2
    from rnagym.config import ConfigFitness

    monkeypatch.setattr(ConfigFitness, "CHECKPOINT_DIR", tmp_path / "ckpt")
    assembled = tmp_path / "ckpt" / "evo2" / "evo2_40b.pt"
    assembled.parent.mkdir(parents=True)
    cache = tmp_path / "verified.json"
    calls: list[str] = []

    def upstream(key: str) -> str:
        calls.append(key)
        if key != "evo2_40b":
            other = tmp_path / f"{key}.pt"
            other.write_bytes(key.encode())
            return str(other)
        if not assembled.exists():
            assembled.write_bytes(b"forty")  # assembled by this call
        return str(assembled)

    monkeypatch.setattr(score_evo2, "evo2_checkpoint", upstream)
    # Another model's check never counts for 40B, even through a symlink
    seven = score_evo2.checkpoint_file("evo2_7b", cache)
    assembled.symlink_to(seven)
    score_evo2.checkpoint_file("evo2_40b", cache)
    assert calls == ["evo2_7b", "evo2_40b"]
    # A file assembled by the check is checked again once, then reused
    assembled.unlink()
    calls.clear()
    for _ in range(3):
        assert score_evo2.checkpoint_file("evo2_40b", cache) == str(assembled)
    assert calls == ["evo2_40b", "evo2_40b"]
    # An edited file is checked again
    assembled.write_bytes(b"edited")
    score_evo2.checkpoint_file("evo2_40b", cache)
    assert calls == ["evo2_40b"] * 3

    def replacing(key: str) -> str:
        calls.append(key)
        assembled.unlink()
        assembled.write_bytes(b"replaced")  # a new file while the old one is checked
        return str(assembled)

    # A file replaced during its check is never recorded
    monkeypatch.setattr(score_evo2, "evo2_checkpoint", replacing)
    assembled.write_bytes(b"edited again")
    calls.clear()
    score_evo2.checkpoint_file("evo2_40b", cache)
    score_evo2.checkpoint_file("evo2_40b", cache)
    assert calls == ["evo2_40b", "evo2_40b"]


def test_package_sources_skip_environments():
    sources = common.package_sources()
    assert Path(common.__file__).resolve() in sources
    assert not any(".pixi" in path.parts for path in sources)
    assert all(path.suffix == ".py" for path in sources)
