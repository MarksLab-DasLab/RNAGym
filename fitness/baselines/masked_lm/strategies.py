"""
The four masked-marginal fill strategies, expressed as contexts and tasks.

Every masked-marginal score computes a log-odds at each mutated position of a
variant and sums over the variant's mutations. The strategies differ in exactly
one thing: what fills the variant's OTHER mutated positions while the scored
position is masked. Following Meier et al. 2021 (ESM-1v) supplement, Appendix A,
with ``M`` the mutated positions, ``x^wt`` and ``x^mt`` the wild-type and variant
sequences, ``x_-i`` a mask at position ``i`` and ``x_-M`` masks at every position
in ``M``:

    wt-fill      sum_i [ log p(mt_i | x^wt_-i) - log p(wt_i | x^wt_-i) ]
    mask-fill    sum_i [ log p(mt_i | x^wt_-M) - log p(wt_i | x^wt_-M) ]
    mut-fill     sum_i [ log p(mt_i | x^mt_-i) - log p(wt_i | x^mt_-i) ]
    match-fill   sum_i [ log p(mt_i | x^mt_-i) - log p(wt_i | x^wt_-i) ]

``wt-fill`` is what the ESM authors' released code and ProteinGym's baseline
implement under the name ``masked-marginals``; ``mask-fill`` is the formula
written in the ESM paper; ``mut-fill`` and ``match-fill`` are strategies c and b
of the supplement. All four are identical on single mutants, because a variant
with one mutation has no other mutated positions and ``x^mt_-i`` equals
``x^wt_-i``. They diverge on multi-mutants, which are 99.4% of this benchmark's
non-coding variants.

Only ``match-fill`` mixes two contexts: its mutant term is conditioned on the
variant's own sequence and its wild-type term on the wild type, so it is a
difference of two conditionals rather than a log-odds ratio.

This module turns an assay into two things:

    contexts   deduplicated strings over the model's alphabet plus a mask
               placeholder, one per unique context example
    tasks      a flat table of (context, position, base, row, strategy, sign)
               terms to accumulate, as parallel arrays

Contexts are shared across strategies wherever they coincide, which is what
makes computing all four cost only about 19% more unique context examples than
computing ``mut-fill`` alone on the non-coding assays. These are context
examples, not model invocations: contexts are batched, and a context carrying
several masks is read at several positions from one pass.

Coordinates
-----------

Three coordinate systems appear and must not be confused:

    full position       0-based index into the assay's sequences, which is what
                        a mutation string names (1-based) and what this module
                        stores in ``TaskTable.pos``
    context position    index into a context string. Equal to the full position
                        unless the context was windowed, in which case
                        ``engine.window_contexts`` subtracts the window origin
                        from ``TaskTable.pos`` so that the stored value is
                        always a context position
    token position      index into the model's input ids, which the adapter
                        computes from the context position, normally by adding
                        the number of leading special tokens

Every task position must point at a mask in its context; ``validate_table``
checks this, because a position that is off by one gathers a neighbouring
nucleotide's distribution and still produces plausible finite scores.
"""

from array import array
from dataclasses import dataclass

import numpy as np
import pandas as pd

MASK_CHAR = "#"

# Canonical order. The CLI accepts the hyphenated spelling used in the write-up;
# the underscored spelling is used for identifiers and output column suffixes.
STRATEGIES = ("wt_fill", "mask_fill", "mut_fill", "match_fill")


def normalize_strategy(name: str) -> str:
    """Accept either the hyphenated or the underscored spelling of a strategy."""
    key = name.strip().lower().replace("-", "_")
    if key not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}; expected one of {STRATEGIES}")
    return key


@dataclass
class TaskTable:
    """
    A deduplicated set of masked contexts plus the terms to read out of them.

    Attributes:
        contexts: masked context strings, deduplicated. A context may
            carry more than one mask (``mask-fill`` masks a variant's whole
            mutated set at once), so a context does not identify the position
            being scored and ``pos`` is carried explicitly.
        ctx_id, pos, base, row, strategy, sign: parallel arrays, one entry per
            term to accumulate. Term ``k`` adds
            ``sign[k] * log p(base[k] | contexts[ctx_id[k]])`` evaluated at
            position ``pos[k]`` into the score of variant ``row[k]`` under
            strategy ``strategy[k]``. Sorted by ``ctx_id`` so that each
            context's terms are a contiguous slice.
        scorable: per-variant mask; False rows are reported as NaN under every
            strategy.
        strategies: the strategy names indexed by ``strategy``.
    """

    contexts: list
    ctx_id: np.ndarray
    pos: np.ndarray
    base: np.ndarray
    row: np.ndarray
    strategy: np.ndarray
    sign: np.ndarray
    scorable: np.ndarray
    strategies: tuple

    def n_terms(self) -> int:
        return int(self.ctx_id.size)


def parse_mutations(mutant_str: str, bases: str) -> list:
    """
    Parse a mutation string such as ``"A4U,A5G"`` into a list of
    ``(pos0, wt_base, mut_base)`` tuples with 0-based positions, in the alphabet
    the model uses.

    Raises:
        ValueError: for non-substitution edits (indels) or bases outside the
        model's alphabet, so the caller can score the affected variant as NaN.
    """
    fold_from, fold_to = ("U", "T") if "T" in bases else ("T", "U")
    mutations = []
    for token in str(mutant_str).replace(" ", "").split(","):
        if not token:
            continue
        wt_base = token[0].upper().replace(fold_from, fold_to)
        mut_base = token[-1].upper().replace(fold_from, fold_to)
        pos = int(token[1:-1]) - 1  # 1-based -> 0-based
        if wt_base not in bases or mut_base not in bases:
            raise ValueError(f"Unsupported mutation token: {token}")
        mutations.append((pos, wt_base, mut_base))
    return mutations


def recover_wild_type(mutants, sequences, bases: str) -> str:
    """
    Recover the assay's wild-type sequence by reverting each variant's own
    mutations, and require that every variant agrees.

    The wild-type-background strategies need a wild type, and taking it from the
    reference sheet alone would not catch a coordinate mismatch between the
    sheet and the assay file. Reverting the mutations uses only the assay, so
    the two are independent and can be cross-checked by the caller.

    Returns:
        The single wild-type sequence implied by the assay.

    Raises:
        ValueError: if the variants do not all imply the same wild type, which
        would mean the assay mixes backgrounds or the mutation coordinates do
        not line up with the sequences.
    """
    candidates = {}
    for mutant_str, seq in zip(mutants, sequences):
        if pd.isna(mutant_str):
            continue
        try:
            mutations = parse_mutations(mutant_str, bases)
        except ValueError:
            continue
        if not mutations:
            continue
        reverted = list(seq)
        ok = True
        for pos, wt_base, mut_base in mutations:
            if pos < 0 or pos >= len(seq) or seq[pos] != mut_base:
                ok = False
                break
            reverted[pos] = wt_base
        if ok:
            candidates.setdefault("".join(reverted), 0)
            candidates["".join(reverted)] += 1
    if not candidates:
        raise ValueError("No variant could be reverted to a wild-type sequence")
    if len(candidates) > 1:
        top = sorted(candidates.items(), key=lambda kv: -kv[1])[:3]
        raise ValueError(
            "Variants imply more than one wild-type sequence "
            f"({len(candidates)} distinct; top counts {[c for _, c in top]})"
        )
    return next(iter(candidates))


def difference_counts(sequences, wild_type: str) -> np.ndarray:
    """
    Count, for every variant, how many positions differ from the wild type.

    Used to verify that a variant differs from the wild type at exactly its
    declared mutated positions and nowhere else. Done as one array comparison
    rather than per row, since the assays run to hundreds of thousands of
    variants.

    Returns:
        int array with one entry per sequence, or -1 where the sequence length
        differs from the wild type's and the comparison is undefined.
    """
    length = len(wild_type)
    counts = np.full(len(sequences), -1, dtype=np.int64)
    same_length = np.array([len(s) == length for s in sequences], dtype=bool)
    if not same_length.any():
        return counts
    packed = np.frombuffer(
        "".join(s for s, keep in zip(sequences, same_length) if keep).encode("ascii"),
        dtype=np.uint8,
    ).reshape(-1, length)
    reference = np.frombuffer(wild_type.encode("ascii"), dtype=np.uint8)
    counts[same_length] = (packed != reference).sum(axis=1)
    return counts


def validate_table(table) -> None:
    """
    Check the task table's invariants: every scored position is masked in the
    context it is read from.

    A position that is off by one, or that survived windowing incorrectly, still
    yields finite and plausible scores, so this is checked rather than assumed.

    Raises:
        ValueError: if any task points at a position that is not masked.
    """
    if table.n_terms() == 0:
        return
    lengths = {len(c) for c in table.contexts}
    if len(lengths) == 1:
        # The usual case: one construct length per assay, so the whole bank
        # packs into an array and the check is one comparison.
        length = lengths.pop()
        packed = np.frombuffer(
            "".join(table.contexts).encode("ascii"), dtype=np.uint8
        ).reshape(len(table.contexts), length)
        if table.pos.min() < 0 or table.pos.max() >= length:
            raise ValueError("A task position falls outside its context")
        bad = packed[table.ctx_id, table.pos] != ord(MASK_CHAR)
        if bad.any():
            k = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"Task {k} reads position {table.pos[k]} of context "
                f"{table.ctx_id[k]}, which is not a mask"
            )
        return
    stride = max(lengths) + 1
    for packed_key in np.unique(table.ctx_id.astype(np.int64) * stride + table.pos):
        ctx, pos = int(packed_key // stride), int(packed_key % stride)
        context = table.contexts[ctx]
        if pos >= len(context) or context[pos] != MASK_CHAR:
            raise ValueError(
                f"Task reads position {pos} of context {ctx}, which is not a mask"
            )


class _ContextBank:
    """Deduplicates context strings and hands out their indices."""

    def __init__(self):
        self._index = {}
        self.contexts = []

    def add(self, context: str) -> int:
        key = self._index.get(context)
        if key is None:
            key = len(self.contexts)
            self._index[context] = key
            self.contexts.append(context)
        return key


def _single_mask(seq: str, pos: int) -> str:
    return seq[:pos] + MASK_CHAR + seq[pos + 1 :]


def _multi_mask(seq: str, positions) -> str:
    chars = list(seq)
    for pos in positions:
        chars[pos] = MASK_CHAR
    return "".join(chars)


def build_tasks(
    mutants,
    sequences,
    wild_type: str,
    bases: str,
    strategies=STRATEGIES,
    verbose: bool = True,
) -> TaskTable:
    """
    Expand an assay's variants into deduplicated masked contexts and the terms
    read out of them, for every requested strategy at once.

    A variant is scorable only if it is scorable under EVERY requested strategy,
    so that the strategies are compared on exactly the same set of variants.

    Every scorable variant is checked to satisfy all of:

        the mutation string parses into substitutions over the alphabet;
        its positions are inside the sequence and are distinct;
        no mutation is a no-op (a mutant base equal to its wild-type base);
        the variant sequence carries the mutant base at each mutated position;
        the sequence contains no mask placeholder;

    and additionally, when a wild-type-background strategy is requested:

        the sequence has the same length as the wild type;
        the wild type carries the declared wild-type base at each position;
        the sequence equals the wild type at every position outside the mutated
        set, so that the declared mutations describe the variant completely.

    Anything else is reported as NaN under every strategy, matching the existing
    scorers' all-or-nothing policy: a variant with one bad mutation contributes
    none of its mutations. On the 31 non-coding assays (856,628 variants) none
    of these checks currently rejects anything, so they cost no coverage.

    Args:
        mutants: the assay's ``mutant`` column.
        sequences: the assay's ``sequence`` column, already folded to the
            model's alphabet.
        wild_type: the assay's wild-type sequence, in the same alphabet.
        bases: the model's alphabet, ``"ACGU"`` or ``"ACGT"``.
        strategies: which strategies to build terms for.
        verbose: print the reason each skipped variant was skipped.

    Returns:
        A TaskTable.
    """
    strategies = tuple(normalize_strategy(s) for s in strategies)
    if not strategies:
        raise ValueError("At least one strategy is required")
    strat_id = {name: i for i, name in enumerate(strategies)}
    base_id = {b: i for i, b in enumerate(bases)}
    needs_wt = any(s in ("wt_fill", "mask_fill", "match_fill") for s in strategies)

    bank = _ContextBank()
    ctx_id, pos_a, row_a = array("i"), array("i"), array("i")
    base_a, strat_a, sign_a = array("b"), array("b"), array("b")
    scorable = np.zeros(len(sequences), dtype=bool)

    # The wild-type single-mask contexts depend only on the position, so they are
    # shared by every variant and worth caching.
    wt_ctx_cache = {}

    def wt_masked(pos: int) -> int:
        key = wt_ctx_cache.get(pos)
        if key is None:
            key = bank.add(_single_mask(wild_type, pos))
            wt_ctx_cache[pos] = key
        return key

    def emit(context_key: int, pos: int, base: str, row: int, strategy: str, sign: int):
        ctx_id.append(context_key)
        pos_a.append(pos)
        base_a.append(base_id[base])
        row_a.append(row)
        strat_a.append(strat_id[strategy])
        sign_a.append(sign)

    # One array comparison for the whole assay, so that the per-variant check
    # that nothing outside the mutated set differs from the wild type is cheap.
    n_differences = difference_counts(sequences, wild_type) if needs_wt else None

    n_skipped = 0
    for i, (mutant_str, seq) in enumerate(zip(mutants, sequences)):
        if pd.isna(mutant_str):
            continue  # wild-type row: leave as NaN
        try:
            mutations = parse_mutations(mutant_str, bases)
            if not mutations:
                continue
            if MASK_CHAR in seq:
                raise ValueError(
                    f"Sequence contains the mask placeholder {MASK_CHAR!r}, which "
                    "would be read as a mask and would corrupt context dedup"
                )
            positions = [pos for pos, _, _ in mutations]
            if len(set(positions)) != len(positions):
                raise ValueError(f"Mutation string names a position twice: {mutant_str}")
            for pos, wt_base, mut_base in mutations:
                if pos < 0 or pos >= len(seq):
                    raise ValueError(f"Mutation position {pos + 1} outside sequence")
                if wt_base == mut_base:
                    raise ValueError(f"Mutation {wt_base}{pos + 1}{mut_base} is a no-op")
                if seq[pos] != mut_base:
                    raise ValueError(
                        f"Sequence has {seq[pos]} at position {pos + 1}, "
                        f"expected the mutant base {mut_base}"
                    )
                if needs_wt:
                    if len(seq) != len(wild_type):
                        raise ValueError(
                            f"Sequence length {len(seq)} differs from the wild type "
                            f"({len(wild_type)}), so wild-type-background strategies "
                            "have no common coordinate frame"
                        )
                    if wild_type[pos] != wt_base:
                        raise ValueError(
                            f"Wild type has {wild_type[pos]} at position {pos + 1}, "
                            f"expected {wt_base}"
                        )
            if needs_wt and n_differences[i] != len(positions):
                raise ValueError(
                    f"Sequence differs from the wild type at {n_differences[i]} "
                    f"positions but declares {len(positions)} mutations, so the "
                    "mutation string does not describe the variant completely"
                )
        except (ValueError, IndexError) as err:
            n_skipped += 1
            if verbose:
                print(f"Skipping variant {mutant_str}: {err}")
            continue  # unsupported edit: leave as NaN under every strategy

        mut_ctx = {}
        if "mut_fill" in strat_id or "match_fill" in strat_id:
            for pos in positions:
                mut_ctx[pos] = bank.add(_single_mask(seq, pos))
        joint_ctx = None
        if "mask_fill" in strat_id:
            joint_ctx = bank.add(_multi_mask(wild_type, positions))

        for pos, wt_base, mut_base in mutations:
            if "wt_fill" in strat_id:
                key = wt_masked(pos)
                emit(key, pos, mut_base, i, "wt_fill", 1)
                emit(key, pos, wt_base, i, "wt_fill", -1)
            if "mask_fill" in strat_id:
                emit(joint_ctx, pos, mut_base, i, "mask_fill", 1)
                emit(joint_ctx, pos, wt_base, i, "mask_fill", -1)
            if "mut_fill" in strat_id:
                emit(mut_ctx[pos], pos, mut_base, i, "mut_fill", 1)
                emit(mut_ctx[pos], pos, wt_base, i, "mut_fill", -1)
            if "match_fill" in strat_id:
                # The mutant term shares mut-fill's context and the wild-type
                # term shares wt-fill's, which is why match-fill is free once
                # both of those are being computed.
                emit(mut_ctx[pos], pos, mut_base, i, "match_fill", 1)
                emit(wt_masked(pos), pos, wt_base, i, "match_fill", -1)
        scorable[i] = True

    if n_skipped and not verbose:
        print(f"Skipped {n_skipped} variants that could not be scored")

    table = TaskTable(
        contexts=bank.contexts,
        ctx_id=np.frombuffer(ctx_id, dtype=np.int32).copy(),
        pos=np.frombuffer(pos_a, dtype=np.int32).copy(),
        base=np.frombuffer(base_a, dtype=np.int8).copy(),
        row=np.frombuffer(row_a, dtype=np.int32).copy(),
        strategy=np.frombuffer(strat_a, dtype=np.int8).copy(),
        sign=np.frombuffer(sign_a, dtype=np.int8).copy(),
        scorable=scorable,
        strategies=strategies,
    )
    order = np.argsort(table.ctx_id, kind="stable")
    table.ctx_id = table.ctx_id[order]
    table.pos = table.pos[order]
    table.base = table.base[order]
    table.row = table.row[order]
    table.strategy = table.strategy[order]
    table.sign = table.sign[order]
    validate_table(table)
    return table
