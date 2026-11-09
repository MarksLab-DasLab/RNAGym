"""Crop a sequence to a model's context around a variant."""

from __future__ import annotations


def crop_bounds(length: int, lo: int, hi: int, budget: int) -> tuple[int, int]:
    """Return ``(start, end)`` of a window of at most ``budget`` positions.

    The arithmetic is that of ``masked_lm.engine.window_contexts``: centre on the
    span ``[lo, hi]``, slide at the sequence ends, never pad, and pass sequences
    that fit through whole.
    """
    if budget < 1:
        raise ValueError(f"budget must be positive, got {budget}")
    if not 0 <= lo <= hi < length:
        raise ValueError(f"span [{lo}, {hi}] is outside a sequence of {length}")
    if length <= budget:
        return 0, length
    if hi - lo + 1 > budget:
        raise ValueError(f"span of {hi - lo + 1} positions exceeds the budget {budget}")
    centre = (lo + hi) // 2
    start = max(0, centre - budget // 2)
    end = min(length, start + budget)
    start = max(0, end - budget)
    return start, end


def variant_span(variant_index: int, ref_allele: str, length: int) -> tuple[int, int]:
    """Return the inclusive span of the reference sequence a crop is centred on.

    An insertion has an empty reference allele and is anchored at its insertion
    point, the base before which it is inserted. An insertion after the last
    base is anchored on that last base.
    """
    lo = min(variant_index, length - 1)
    return lo, max(lo, variant_index + len(ref_allele) - 1)
