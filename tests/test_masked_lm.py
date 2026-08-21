"""
Unit tests for the shared masked-marginal scoring engine.

These run on CPU against a small deterministic stand-in model, so they check the
four fill strategies, the context bank and the accumulation without needing a
checkpoint or a GPU. The stand-in's logits at a position depend on the whole
context, which is what makes the four strategies disagree on multi-mutants, as
they must.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fitness" / "baselines"))

from masked_lm import (  # noqa: E402
    STRATEGIES,
    MaskedLMAdapter,
    accumulate_scores,
    build_tasks,
)
from masked_lm.strategies import MASK_CHAR, recover_wild_type  # noqa: E402

VOCAB = {"<pad>": 0, "<cls>": 1, "<eos>": 2, "<mask>": 3, "A": 4, "C": 5, "G": 6, "U": 7, "N": 8}


class FakeAdapter(MaskedLMAdapter):
    """A deterministic stand-in whose logits depend on the entire context."""

    name = "fake"
    bases = "ACGU"
    score_column = "fake_score"
    n_special_tokens = 2  # <cls> and <eos>

    def __init__(self):
        self.device = "cpu"
        self.prefix_ids = [VOCAB["<cls>"]]
        self.suffix_ids = [VOCAB["<eos>"]]
        self.pad_id = VOCAB["<pad>"]
        self.mask_id = VOCAB["<mask>"]
        self.unk_id = VOCAB["N"]
        self.base_ids = {b: VOCAB[b] for b in self.bases}
        generator = torch.Generator().manual_seed(0)
        self.embedding = torch.randn(len(VOCAB), 8, generator=generator)
        self.projection = torch.randn(8, len(VOCAB), generator=generator)

    @staticmethod
    def add_arguments(parser):
        pass

    def load(self, args):
        pass

    def _logits(self, input_ids):
        embedded = self.embedding[input_ids]
        # Every position sees the whole context, so filling the other mutated
        # positions differently changes the prediction, as in a real masked LM.
        pooled = embedded.mean(dim=1, keepdim=True)
        return (embedded + pooled) @ self.projection

    def logits_at(self, input_ids, attention_mask, rows, cols):
        return self._logits(input_ids)[rows, cols]

    def naive_log_probs(self, context: str, position: int):
        """Reference path: encode one context on its own and read one position."""
        ids = list(self.prefix_ids)
        for char in context:
            ids.append(self.mask_id if char == MASK_CHAR else self.base_ids[char])
        ids.extend(self.suffix_ids)
        logits = self._logits(torch.tensor([ids], dtype=torch.long))
        return torch.log_softmax(logits[0, position + len(self.prefix_ids)].float(), -1).numpy()


def naive_score(adapter, wild_type, sequence, mutations, strategy):
    """
    Score one variant straight from the formulas, one context at a time.

    Deliberately independent of the context bank and the batching engine.
    """
    total = 0.0
    positions = [pos for pos, _, _ in mutations]
    for pos, wt_base, mut_base in mutations:
        if strategy == "wt_fill":
            ctx = wild_type[:pos] + MASK_CHAR + wild_type[pos + 1 :]
            lp = adapter.naive_log_probs(ctx, pos)
            total += lp[adapter.base_ids[mut_base]] - lp[adapter.base_ids[wt_base]]
        elif strategy == "mask_fill":
            chars = list(wild_type)
            for other in positions:
                chars[other] = MASK_CHAR
            lp = adapter.naive_log_probs("".join(chars), pos)
            total += lp[adapter.base_ids[mut_base]] - lp[adapter.base_ids[wt_base]]
        elif strategy == "mut_fill":
            ctx = sequence[:pos] + MASK_CHAR + sequence[pos + 1 :]
            lp = adapter.naive_log_probs(ctx, pos)
            total += lp[adapter.base_ids[mut_base]] - lp[adapter.base_ids[wt_base]]
        elif strategy == "match_fill":
            mut_ctx = sequence[:pos] + MASK_CHAR + sequence[pos + 1 :]
            wt_ctx = wild_type[:pos] + MASK_CHAR + wild_type[pos + 1 :]
            total += adapter.naive_log_probs(mut_ctx, pos)[adapter.base_ids[mut_base]]
            total -= adapter.naive_log_probs(wt_ctx, pos)[adapter.base_ids[wt_base]]
        else:
            raise ValueError(strategy)
    return total


def make_assay(n_variants, max_mutations, length=24, seed=0):
    """Build a synthetic assay: a wild type plus random substitution variants."""
    rng = np.random.default_rng(seed)
    bases = list("ACGU")
    wild_type = "".join(rng.choice(bases, size=length))
    mutants, sequences, parsed = [], [], []
    for _ in range(n_variants):
        k = int(rng.integers(1, max_mutations + 1))
        positions = sorted(rng.choice(length, size=k, replace=False).tolist())
        tokens, seq = [], list(wild_type)
        muts = []
        for pos in positions:
            wt_base = wild_type[pos]
            mut_base = str(rng.choice([b for b in bases if b != wt_base]))
            seq[pos] = mut_base
            tokens.append(f"{wt_base}{pos + 1}{mut_base}")
            muts.append((pos, wt_base, mut_base))
        mutants.append(",".join(tokens))
        sequences.append("".join(seq))
        parsed.append(muts)
    return wild_type, mutants, sequences, parsed


def score_with_engine(adapter, wild_type, mutants, sequences, strategies):
    table = build_tasks(mutants, sequences, wild_type, adapter.bases, strategies, verbose=False)
    scores = accumulate_scores(
        adapter, table, n_rows=len(sequences), batch_size=7, max_batch_tokens=10**6,
        progress=False,
    )
    return table, scores


@pytest.fixture(scope="module")
def adapter():
    return FakeAdapter()


def test_matches_the_formulas(adapter):
    """Every strategy reproduces a per-variant, one-context-at-a-time reference."""
    wild_type, mutants, sequences, parsed = make_assay(40, 4, seed=1)
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    _, scores = score_with_engine(adapter, wild_type, mutants, sequences, strategies)
    for s, strategy in enumerate(strategies):
        expected = [
            naive_score(adapter, wild_type, seq, muts, strategy)
            for seq, muts in zip(sequences, parsed)
        ]
        assert np.allclose(scores[s], expected, atol=1e-6), strategy


def test_single_mutants_are_identical(adapter):
    """
    With one mutation there are no other mutated positions, so the four
    strategies are the same computation. This is the Pitt_2010_ribozyme fixture
    in miniature: disagreement here means a bug.
    """
    wild_type, mutants, sequences, _ = make_assay(30, 1, seed=2)
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    _, scores = score_with_engine(adapter, wild_type, mutants, sequences, strategies)
    assert np.allclose(scores, scores[0], atol=1e-9)


def test_multi_mutants_diverge(adapter):
    """The strategies must not silently collapse onto one another."""
    wild_type, mutants, sequences, _ = make_assay(30, 4, seed=3)
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    _, scores = score_with_engine(adapter, wild_type, mutants, sequences, strategies)
    for i in range(1, 4):
        assert not np.allclose(scores[i], scores[0])


def test_contexts_are_shared_across_strategies(adapter):
    """
    Asking for all four costs far less than asking for each separately, because
    match-fill reuses mut-fill's and wt-fill's contexts and single mutants make
    mask-fill's context equal to wt-fill's.
    """
    wild_type, mutants, sequences, _ = make_assay(30, 4, seed=4)
    sizes = {}
    for strategy in ("wt_fill", "mask_fill", "mut_fill", "match_fill"):
        table, _ = score_with_engine(adapter, wild_type, mutants, sequences, (strategy,))
        sizes[strategy] = len(table.contexts)
    joint, _ = score_with_engine(
        adapter, wild_type, mutants, sequences, ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    )
    assert len(joint.contexts) < sum(sizes.values())
    # match-fill introduces no context that mut-fill and wt-fill do not already need
    assert sizes["match_fill"] <= sizes["mut_fill"] + sizes["wt_fill"]
    combined, _ = score_with_engine(
        adapter, wild_type, mutants, sequences, ("wt_fill", "mut_fill", "match_fill")
    )
    pair, _ = score_with_engine(adapter, wild_type, mutants, sequences, ("wt_fill", "mut_fill"))
    assert len(combined.contexts) == len(pair.contexts)


def test_strategy_subsets_agree_with_the_full_run(adapter):
    """Requesting one strategy gives the same numbers as requesting all four."""
    wild_type, mutants, sequences, _ = make_assay(25, 3, seed=5)
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    _, full = score_with_engine(adapter, wild_type, mutants, sequences, strategies)
    for i, strategy in enumerate(strategies):
        _, alone = score_with_engine(adapter, wild_type, mutants, sequences, (strategy,))
        assert np.allclose(full[i], alone[0], atol=1e-9), strategy


def test_mask_fill_masks_the_whole_mutated_set(adapter):
    """mask-fill uses one context per variant, carrying |M| masks."""
    wild_type = "ACGUACGUACGU"
    mutants = ["A1C,G3U"]
    sequences = ["CCUUACGUACGU"]
    table = build_tasks(mutants, sequences, wild_type, "ACGU", ("mask_fill",), verbose=False)
    assert len(table.contexts) == 1
    assert table.contexts[0].count(MASK_CHAR) == 2
    assert sorted(table.pos.tolist()) == [0, 0, 2, 2]


def test_wt_and_mut_fill_mask_one_position(adapter):
    """wt-fill masks one position of the wild type; mut-fill masks one position
    of the variant, so the variant's other mutation stays visible."""
    wild_type = "ACGUACGUACGU"
    mutants = ["A1C,G3U"]
    sequences = ["CCUUACGUACGU"]

    wt_table = build_tasks(mutants, sequences, wild_type, "ACGU", ("wt_fill",), verbose=False)
    assert sorted(wt_table.contexts) == sorted(["#CGUACGUACGU", "AC#UACGUACGU"])

    mut_table = build_tasks(mutants, sequences, wild_type, "ACGU", ("mut_fill",), verbose=False)
    assert sorted(mut_table.contexts) == sorted(["#CUUACGUACGU", "CC#UACGUACGU"])


def test_task_records_carry_the_right_base_sign_and_strategy(adapter):
    """
    The term table is where a silent mix-up would be invisible, so check every
    field of a two-mutation variant under all four strategies by hand.
    """
    wild_type = "ACGUACGUACGU"
    mutants = ["A1C,G3U"]
    sequences = ["CCUUACGUACGU"]
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    table = build_tasks(mutants, sequences, wild_type, "ACGU", strategies, verbose=False)

    bases = "ACGU"
    seen = set()
    for k in range(table.n_terms()):
        seen.add(
            (
                table.contexts[table.ctx_id[k]],
                int(table.pos[k]),
                bases[table.base[k]],
                strategies[table.strategy[k]],
                int(table.sign[k]),
            )
        )
    assert seen == {
        # wt-fill: both alleles from the wild-type context, one mask
        ("#CGUACGUACGU", 0, "C", "wt_fill", 1),
        ("#CGUACGUACGU", 0, "A", "wt_fill", -1),
        ("AC#UACGUACGU", 2, "U", "wt_fill", 1),
        ("AC#UACGUACGU", 2, "G", "wt_fill", -1),
        # mask-fill: both alleles from one wild-type context masked at both sites
        ("#C#UACGUACGU", 0, "C", "mask_fill", 1),
        ("#C#UACGUACGU", 0, "A", "mask_fill", -1),
        ("#C#UACGUACGU", 2, "U", "mask_fill", 1),
        ("#C#UACGUACGU", 2, "G", "mask_fill", -1),
        # mut-fill: both alleles from the variant's own context
        ("#CUUACGUACGU", 0, "C", "mut_fill", 1),
        ("#CUUACGUACGU", 0, "A", "mut_fill", -1),
        ("CC#UACGUACGU", 2, "U", "mut_fill", 1),
        ("CC#UACGUACGU", 2, "G", "mut_fill", -1),
        # match-fill: mutant allele from the variant context, wild-type allele
        # from the wild-type context
        ("#CUUACGUACGU", 0, "C", "match_fill", 1),
        ("#CGUACGUACGU", 0, "A", "match_fill", -1),
        ("CC#UACGUACGU", 2, "U", "match_fill", 1),
        ("AC#UACGUACGU", 2, "G", "match_fill", -1),
    }


def test_every_scored_position_is_masked(adapter):
    """The table's own invariant check must reject a position that is not a mask."""
    from masked_lm.strategies import validate_table

    wild_type, mutants, sequences, _ = make_assay(10, 3, seed=8)
    table = build_tasks(mutants, sequences, wild_type, "ACGU", ("mask_fill",), verbose=False)
    validate_table(table)
    table.pos = table.pos + 1  # shift every read off its mask
    with pytest.raises(ValueError):
        validate_table(table)


def test_multi_mask_context_is_gathered_at_every_masked_position(adapter):
    """
    mask-fill reads one context at several positions. The engine must gather
    each of them, not just the first.
    """
    wild_type = "ACGUACGUACGU"
    mutants = ["A1C,G3U"]
    sequences = ["CCUUACGUACGU"]
    table = build_tasks(mutants, sequences, wild_type, "ACGU", ("mask_fill",), verbose=False)
    assert len(table.contexts) == 1
    scores = accumulate_scores(
        adapter, table, n_rows=1, batch_size=4, max_batch_tokens=10**6, progress=False
    )
    context = table.contexts[0]
    expected = 0.0
    for pos, wt_base, mut_base in ((0, "A", "C"), (2, "G", "U")):
        lp = adapter.naive_log_probs(context, pos)
        expected += lp[adapter.base_ids[mut_base]] - lp[adapter.base_ids[wt_base]]
    assert np.isclose(scores[0, 0], expected, atol=1e-6)


def test_prefix_offset_is_applied(adapter):
    """
    A model with leading special tokens must read the token that carries the
    nucleotide, not the special token. Scoring the same context with and without
    a prefix must agree.
    """
    wild_type, mutants, sequences, _ = make_assay(12, 2, seed=9)
    table = build_tasks(mutants, sequences, wild_type, "ACGU", ("mut_fill",), verbose=False)
    with_prefix = accumulate_scores(
        adapter, table, len(sequences), batch_size=8, max_batch_tokens=10**6, progress=False
    )

    bare = FakeAdapter()
    bare.prefix_ids = []
    bare.suffix_ids = []
    without_prefix = accumulate_scores(
        bare, table, len(sequences), batch_size=8, max_batch_tokens=10**6, progress=False
    )
    # Different inputs, so different numbers, but both must be finite and the
    # offset must have moved with the prefix rather than staying at zero.
    assert np.isfinite(with_prefix).all() and np.isfinite(without_prefix).all()
    assert not np.allclose(with_prefix, without_prefix)

    shifted = FakeAdapter()
    shifted.token_position = lambda pos: pos  # forget the prefix offset
    wrong = accumulate_scores(
        shifted, table, len(sequences), batch_size=8, max_batch_tokens=10**6, progress=False
    )
    assert not np.allclose(with_prefix, wrong)


def test_unknown_context_bases_are_encoded_as_unknown(adapter):
    """
    A construct may contain an N. It is encoded as the unknown token in the
    context, while mutation alleles themselves must be canonical bases.
    """
    wild_type = "ACGUNCGUACGU"
    mutants = ["A1C", "N5C"]
    sequences = ["CCGUNCGUACGU", "ACGUCCGUACGU"]
    table = build_tasks(mutants, sequences, wild_type, "ACGU", ("mut_fill",), verbose=False)
    assert table.scorable.tolist() == [True, False]  # N is not a scorable allele
    ids = adapter.encode_context("ACGUNCGU", MASK_CHAR)
    assert ids[5] == adapter.unk_id  # position 4 plus the one leading token


def test_mask_placeholder_in_a_sequence_is_rejected(adapter):
    """A sequence carrying the placeholder would silently become a mask."""
    wild_type = "ACGUACGUACGU"
    table = build_tasks(
        ["A1C"], ["CCGUACGUAC" + MASK_CHAR + "U"], wild_type, "ACGU", ("mut_fill",), verbose=False
    )
    assert not table.scorable.any()


def test_malformed_variants_are_rejected(adapter):
    """Duplicate positions, no-ops, and undeclared changes are all rejected."""
    wild_type = "ACGUACGUACGU"
    cases = [
        ("A1C,A1G", "GCGUACGUACGU"),          # same position named twice
        ("A1A", "ACGUACGUACGU"),              # no-op mutation
        ("A1C", "CCGUACGUACGA"),              # a change the mutation string omits
    ]
    for mutant, sequence in cases:
        table = build_tasks([mutant], [sequence], wild_type, "ACGU", STRATEGIES, verbose=False)
        assert not table.scorable.any(), mutant


def test_unscorable_variants_are_nan_under_every_strategy(adapter):
    """A row that any strategy cannot score is NaN for all of them, so the
    strategies are compared on exactly the same variants."""
    wild_type = "ACGUACGUACGU"
    mutants = [
        "A1C",          # fine
        "A99C",         # position outside the sequence
        "A1-",          # not a substitution
        "G1C",          # wild-type base disagrees with the wild type
        None,           # the wild-type row itself
    ]
    sequences = ["CCGUACGUACGU", "ACGUACGUACGU", "ACGUACGUACGU", "CCGUACGUACGU", wild_type]
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    table, scores = score_with_engine(adapter, wild_type, mutants, sequences, strategies)
    assert table.scorable.tolist() == [True, False, False, False, False]
    assert np.isfinite(scores[:, 0]).all()
    assert np.isnan(scores[:, 1:]).all()


def test_wild_type_recovery():
    """The wild type is recoverable from the variants alone, and disagreement is
    an error rather than a silently chosen majority."""
    wild_type, mutants, sequences, _ = make_assay(20, 3, seed=6)
    assert recover_wild_type(mutants, sequences, "ACGU") == wild_type
    broken = list(sequences)
    broken[0] = "A" * len(wild_type)
    mutants = list(mutants)
    mutants[0] = "A1A"
    with pytest.raises(ValueError):
        recover_wild_type(mutants, broken, "ACGU")


def test_batching_does_not_change_scores(adapter):
    """Scores must not depend on how contexts are packed into batches."""
    wild_type, mutants, sequences, _ = make_assay(25, 3, seed=7)
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    table = build_tasks(mutants, sequences, wild_type, "ACGU", strategies, verbose=False)
    reference = accumulate_scores(
        adapter, table, len(sequences), batch_size=1, max_batch_tokens=10**6, progress=False
    )
    for batch_size in (3, 16, 1000):
        other = accumulate_scores(
            adapter, table, len(sequences), batch_size=batch_size,
            max_batch_tokens=10**6, progress=False,
        )
        assert np.allclose(reference, other, atol=1e-10), batch_size


def test_shared_distribution_feeds_several_rows_and_strategies(adapter):
    """
    One gathered distribution is read by several variants, strategies and signs.
    A flat-index or accumulation error would show up as one row stealing
    another's contribution, so check the accumulation against a hand sum.
    """
    wild_type = "ACGUACGUACGU"
    # All three variants mutate position 0, so wt-fill reads one shared context
    # there; the first two also share their mut-fill context at position 0.
    mutants = ["A1C", "A1C,G3U", "A1G"]
    sequences = ["CCGUACGUACGU", "CCUUACGUACGU", "GCGUACGUACGU"]
    strategies = ("wt_fill", "mut_fill", "match_fill")
    table, scores = score_with_engine(adapter, wild_type, mutants, sequences, strategies)

    shared = "#CGUACGUACGU"
    assert shared in table.contexts
    lp = adapter.naive_log_probs(shared, 0)
    # every wt-fill term at position 0 comes from that one distribution
    assert np.isclose(scores[0, 0], lp[adapter.base_ids["C"]] - lp[adapter.base_ids["A"]])
    assert np.isclose(scores[0, 2], lp[adapter.base_ids["G"]] - lp[adapter.base_ids["A"]])
    for s, strategy in enumerate(strategies):
        expected = [
            naive_score(adapter, wild_type, seq, muts, strategy)
            for seq, muts in zip(sequences, [
                [(0, "A", "C")],
                [(0, "A", "C"), (2, "G", "U")],
                [(0, "A", "G")],
            ])
        ]
        assert np.allclose(scores[s], expected, atol=1e-6), strategy


def test_gather_key_packing_at_the_boundaries(adapter):
    """
    The engine packs (context, position) into one integer to find unique
    distributions. Check the corners: the last position of one context and the
    first position of the next must not collide.
    """
    wild_type = "ACGUACGUACGU"
    length = len(wild_type)
    # one variant mutating the final position, one mutating the first
    mutants = [f"U{length}A", "A1C"]
    sequences = ["ACGUACGUACGA", "CCGUACGUACGU"]
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    _, scores = score_with_engine(adapter, wild_type, mutants, sequences, strategies)
    expected_last = naive_score(adapter, wild_type, sequences[0], [(length - 1, "U", "A")], "mut_fill")
    expected_first = naive_score(adapter, wild_type, sequences[1], [(0, "A", "C")], "mut_fill")
    assert np.isclose(scores[2, 0], expected_last, atol=1e-6)
    assert np.isclose(scores[2, 1], expected_first, atol=1e-6)
    assert not np.isclose(expected_last, expected_first)


def test_batch_boundary_falls_around_a_heavily_used_context(adapter):
    """
    Terms are sliced per batch with searchsorted, so a context carrying many
    terms must be handled whichever side of a batch boundary it lands on.
    """
    wild_type = "ACGUACGUACGU"
    # 20 variants all mutating position 0, so one wt-fill context carries 40 terms
    mutants, sequences = [], []
    for i in range(20):
        other = 2 + (i % 8)
        wt_other = wild_type[other]
        mut_other = "A" if wt_other != "A" else "C"
        mutants.append(f"A1C,{wt_other}{other + 1}{mut_other}")
        seq = list(wild_type)
        seq[0], seq[other] = "C", mut_other
        sequences.append("".join(seq))
    strategies = ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    table = build_tasks(mutants, sequences, wild_type, "ACGU", strategies, verbose=False)
    reference = accumulate_scores(
        adapter, table, len(sequences), batch_size=len(table.contexts),
        max_batch_tokens=10**6, progress=False,
    )
    for batch_size in range(1, 6):
        chunked = accumulate_scores(
            adapter, table, len(sequences), batch_size=batch_size,
            max_batch_tokens=10**6, progress=False,
        )
        assert np.allclose(reference, chunked, atol=1e-10), batch_size


def test_windowing_keeps_masks_and_positions_consistent(adapter):
    """
    Windowing shifts positions. Masks near both ends of a long construct must
    survive it, and the table's invariant must still hold afterwards.
    """
    from masked_lm.engine import window_contexts
    from masked_lm.strategies import validate_table

    length = 200
    rng = np.random.default_rng(11)
    wild_type = "".join(rng.choice(list("ACGU"), size=length))
    mutants, sequences = [], []
    for pos in (1, length - 2):
        wt_base = wild_type[pos]
        mut_base = "A" if wt_base != "A" else "C"
        seq = list(wild_type)
        seq[pos] = mut_base
        mutants.append(f"{wt_base}{pos + 1}{mut_base}")
        sequences.append("".join(seq))
    table = build_tasks(mutants, sequences, wild_type, "ACGU", ("mut_fill",), verbose=False)
    window_contexts(table, 64)
    validate_table(table)
    assert all(len(c) <= 64 for c in table.contexts)
    scores = accumulate_scores(
        adapter, table, len(sequences), batch_size=2, max_batch_tokens=10**6, progress=False
    )
    assert np.isfinite(scores).all()


def test_window_guard_uses_the_declared_special_token_count():
    """
    The guard that refuses multi-strategy scoring on over-long constructs runs
    before the model is loaded, so it must use the adapter's declared special
    token count. Reading the unloaded prefix_ids instead would give a budget two
    positions too large and let a 1023 nt construct through.
    """
    from masked_lm.runner import needs_windowing, window_budget

    unloaded = FakeAdapter.__new__(FakeAdapter)  # not loaded: no prefix_ids yet
    unloaded.max_tokens = 1024
    assert window_budget(unloaded, 1024) == 1022
    assert needs_windowing(["A" * 1023], window_budget(unloaded, 1024))
    assert not needs_windowing(["A" * 1022], window_budget(unloaded, 1024))


@pytest.mark.parametrize(
    "module_dir,module_name,expect",
    [
        ("RNA_FM", "score_rna_fm_single_dms", {"cls": "RNAFMAdapter", "bases": "ACGU",
         "column": "RNA_FM_scores", "special": 2, "batch": 512, "tokens": 65536, "dtype": None}),
        ("RiNALMo", "score_rinalmo_single_dms", {"cls": "RiNALMoAdapter", "bases": "ACGT",
         "column": "logit_scores", "special": 2, "batch": 512, "tokens": 65536, "dtype": None}),
        ("AIDO_RNA", "score_aido_rna_single_dms", {"cls": "AIDORNAAdapter", "bases": "ACGT",
         "column": "aido_rna_score", "special": 2, "batch": 512, "tokens": 49152,
         "dtype": "bfloat16"}),
        ("RNAGenesis", "score_rnagenesis_single_dms", {"cls": "RNAGenesisAdapter", "bases": "ACGU",
         "column": "rnagenesis_score", "special": 0, "batch": 256, "tokens": 32768,
         "dtype": "bfloat16"}),
    ],
)
def test_adapters_keep_their_historical_defaults(module_dir, module_name, expect):
    """
    Alphabet, output column, batching and dtype defaults are load bearing: they
    are what the shipped predictions were produced with. A refactor that changes
    one silently changes the benchmark's numbers.
    """
    import argparse
    import importlib.util

    path = (
        Path(__file__).resolve().parent.parent
        / "fitness" / "baselines" / module_dir / f"{module_name}.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = getattr(module, expect["cls"])()

    assert adapter.bases == expect["bases"]
    assert adapter.score_column == expect["column"]
    assert adapter.n_special_tokens == expect["special"]
    assert adapter.default_batch_size == expect["batch"]
    assert adapter.default_max_batch_tokens == expect["tokens"]

    parser = argparse.ArgumentParser()
    adapter.add_arguments(parser)
    defaults = {a.dest: a.default for a in parser._actions}
    assert defaults.get("dtype") == expect["dtype"]


def test_performance_fitness_metrics_are_directed():
    """
    All three benchmark metrics report direction: a model that ranks variants
    the wrong way round must score worse than random, not the same as a model
    that ranks them correctly. Spearman was once an absolute value, the AUC was
    folded with max(auc, 1 - auc) and the MCC was absolute, which let one row
    call a model badly wrong and moderately good at the same time. Pinned here
    rather than left to a reviewer to notice again.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fitness.performance_fitness import calculate_metrics

    truth = np.arange(40, dtype=float)
    perfect = calculate_metrics(truth, truth)
    backwards = calculate_metrics(truth, -truth)
    assert perfect["Spearman"] == pytest.approx(1.0)
    assert backwards["Spearman"] == pytest.approx(-1.0)
    assert perfect["AUC"] == pytest.approx(1.0)
    assert backwards["AUC"] == pytest.approx(0.0)
    assert perfect["MCC"] == pytest.approx(1.0)
    assert backwards["MCC"] == pytest.approx(-1.0)

    rng = np.random.default_rng(0)
    noisy = truth + rng.normal(0, 5, truth.size)
    forward = calculate_metrics(truth, noisy)
    reversed_ = calculate_metrics(truth, -noisy)
    assert forward["Spearman"] == pytest.approx(-reversed_["Spearman"])
    assert forward["AUC"] > 0.5 > reversed_["AUC"]
    assert forward["MCC"] > 0 > reversed_["MCC"]
