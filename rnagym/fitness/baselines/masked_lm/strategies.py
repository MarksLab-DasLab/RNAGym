"""Build shared contexts and score terms for masked-marginal strategies.

The formulas and provenance are documented in this package's README. Task
positions are zero-based context coordinates and must point at a mask.
"""

from array import array
from dataclasses import dataclass

import numpy as np
import pandas as pd

MASK_CHAR = "#"

# Canonical order. The CLI accepts the hyphenated spelling used in the write-up
# the underscored spelling is used for identifiers and output column suffixes
STRATEGIES = ("wt_fill", "mask_fill", "mut_fill", "match_fill")


def normalize_strategy(name: str) -> str:
    """Accept either the hyphenated or the underscored spelling of a strategy."""
    key = name.strip().lower().replace("-", "_")
    if key not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}. Expected one of {STRATEGIES}")
    return key


@dataclass
class TaskTable:
    """Deduplicated contexts and parallel arrays of accumulation terms."""

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
    """Return ``(position, wild base, mutant base)`` substitution tuples."""
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
    """Recover one wild type by reverting every valid variant."""
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
            f"({len(candidates)} distinct, top counts {[c for _, c in top]})"
        )
    return next(iter(candidates))


def difference_counts(sequences, wild_type: str) -> np.ndarray:
    """Count differences, using -1 for sequences of the wrong length."""
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
    """Require every task position to point at a mask in its context."""
    if table.n_terms() == 0:
        return
    if table.ctx_id.min() < 0 or table.ctx_id.max() >= len(table.contexts):
        raise ValueError("A task references an unknown context")
    context_lengths = np.fromiter(
        (len(context) for context in table.contexts), dtype=np.int64
    )
    outside = (table.pos < 0) | (table.pos >= context_lengths[table.ctx_id])
    if outside.any():
        raise ValueError("A task position falls outside its context")

    lengths = {len(c) for c in table.contexts}
    if len(lengths) == 1:
        # The usual case: one construct length per assay, so the whole bank
        # packs into an array and the check is one comparison
        length = lengths.pop()
        packed = np.frombuffer(
            "".join(table.contexts).encode("ascii"), dtype=np.uint8
        ).reshape(len(table.contexts), length)
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
    """Build deduplicated contexts and signed log-probability terms.

    Invalid variants remain NaN under every requested strategy. Wild-type
    strategies also require the mutation string to describe every difference
    from the supplied wild type.
    """
    strategies = tuple(normalize_strategy(s) for s in strategies)
    if not strategies:
        raise ValueError("At least one strategy is required")
    if len(set(strategies)) != len(strategies):
        raise ValueError(f"Duplicate strategies requested: {strategies}")
    strat_id = {name: i for i, name in enumerate(strategies)}
    base_id = {b: i for i, b in enumerate(bases)}
    needs_wt = any(s in ("wt_fill", "mask_fill", "match_fill") for s in strategies)

    bank = _ContextBank()
    ctx_id, pos_a, row_a = array("i"), array("i"), array("i")
    base_a, strat_a, sign_a = array("b"), array("b"), array("b")
    scorable = np.zeros(len(sequences), dtype=bool)

    # The wild-type single-mask contexts depend only on the position, so they are
    # shared by every variant and worth caching
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
    # that nothing outside the mutated set differs from the wild type is cheap
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
                raise ValueError(
                    f"Mutation string names a position twice: {mutant_str}"
                )
            for pos, wt_base, mut_base in mutations:
                if pos < 0 or pos >= len(seq):
                    raise ValueError(f"Mutation position {pos + 1} outside sequence")
                if wt_base == mut_base:
                    raise ValueError(
                        f"Mutation {wt_base}{pos + 1}{mut_base} is a no-op"
                    )
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
                # both of those are being computed
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
