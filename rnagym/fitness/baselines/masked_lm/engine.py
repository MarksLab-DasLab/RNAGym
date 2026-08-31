"""Run batched inference over deduplicated masked contexts."""

import numpy as np
import torch
from rnagym.fitness.baselines.masked_lm.strategies import MASK_CHAR, validate_table
from tqdm.auto import tqdm


def window_contexts(table, budget: int) -> None:
    """Window contexts in place around their masked span."""
    if all(len(c) <= budget for c in table.contexts):
        return

    # Masked span per context, from the task table rather than from the string
    # The table is sorted by context, so the spans come from one grouped reduce
    # rather than a pass over every term, which matters on the mRNA-coding
    # assays where this path is reached at all
    span_lo = {}
    span_hi = {}
    if table.n_terms():
        used, first = np.unique(table.ctx_id, return_index=True)
        lows = np.minimum.reduceat(table.pos, first)
        highs = np.maximum.reduceat(table.pos, first)
        span_lo = dict(zip(used.tolist(), lows.tolist()))
        span_hi = dict(zip(used.tolist(), highs.tolist()))

    shift = np.zeros(len(table.contexts), dtype=np.int64)
    trimmed = list(table.contexts)
    for k, context in enumerate(table.contexts):
        if len(context) <= budget:
            continue
        if k not in span_lo:
            trimmed[k] = context[:budget]
            continue
        low, high = span_lo[k], span_hi[k]
        span = high - low + 1
        if span > budget:
            raise ValueError(
                f"Masked span of {span} positions does not fit in a window of "
                f"{budget}. This context cannot be scored"
            )
        centre = (low + high) // 2
        start = max(0, centre - budget // 2)
        end = min(len(context), start + budget)
        start = max(0, end - budget)
        trimmed[k] = context[start:end]
        shift[k] = start

    table.contexts = trimmed
    if shift.any():
        table.pos = (table.pos - shift[table.ctx_id]).astype(np.int32)
    validate_table(table)


def pad_contexts(table, adapter) -> None:
    """Pad fixed-length contexts to the adapter's required length."""
    if adapter.context_pad_char is None:
        return
    lengths = {len(c) for c in table.contexts}
    if len(lengths) != 1:
        raise ValueError(
            f"{adapter.name} pads contexts to a fixed length, so every context "
            f"must start the same length. Found {sorted(lengths)}"
        )
    length = lengths.pop()
    target = adapter.context_length_for(length)
    if target < length:
        raise ValueError(
            f"{adapter.name} asked to pad a {length} nt context down to {target}"
        )
    if target == length:
        return
    padding = target - length
    filler = adapter.context_pad_char
    print(
        f"Padding contexts from {length} to {target} positions with "
        f"{padding} trailing {filler}"
    )
    table.contexts = [context + filler * padding for context in table.contexts]
    validate_table(table)


def accumulate_scores(
    adapter,
    table,
    n_rows: int,
    batch_size: int,
    max_batch_tokens: int,
    progress: bool = True,
) -> np.ndarray:
    """Return one score row per strategy, leaving unscorable variants NaN."""
    contexts = table.contexts
    if not contexts:
        raise ValueError("No scorable variants found")

    n_special = len(adapter.prefix_ids) + len(adapter.suffix_ids)
    # The gather is vectorized, so it needs the context-to-token map to be a
    # constant shift. The adapter declares that, and the declaration is checked
    # here: an off-by-one reads a neighbouring nucleotide's distribution and
    # still produces plausible finite scores
    if not adapter.constant_token_offset:
        raise ValueError(
            f"{adapter.name} does not declare a constant context-to-token offset, "
            "so the engine cannot gather its logits by a vectorized shift"
        )
    offset = adapter.token_position(0)
    longest = max(len(c) for c in contexts)
    if any(adapter.token_position(p) != offset + p for p in (0, 1, longest - 1)):
        raise ValueError(
            f"{adapter.name} declares a constant context-to-token offset but does "
            "not implement one"
        )
    vocab_of_base = np.array(
        [adapter.base_ids[b] for b in adapter.bases], dtype=np.int64
    )

    seq_len = max(len(c) for c in contexts)
    encoded_length = seq_len + n_special
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_batch_tokens < encoded_length:
        raise ValueError(
            f"max_batch_tokens={max_batch_tokens} cannot hold one encoded context "
            f"of length {encoded_length}"
        )
    batch_size = min(batch_size, max_batch_tokens // encoded_length)
    print(f"Using batch size {batch_size} for sequence length {seq_len}")

    scores = np.full((len(table.strategies), n_rows), np.nan, dtype=float)
    scores[:, table.scorable] = 0.0
    flat = scores.reshape(-1)

    device = adapter.device
    pad_id = adapter.pad_id

    for start in tqdm(
        range(0, len(contexts), batch_size),
        desc="Scoring",
        unit="batch",
        disable=not progress,
    ):
        batch = contexts[start : start + batch_size]
        max_len = max(len(c) for c in batch) + n_special

        if not adapter.allows_mixed_length_batches and len({len(c) for c in batch}) > 1:
            raise ValueError(
                f"{adapter.name} is not declared padding invariant, so contexts of "
                "different lengths must not share a batch"
            )

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for b, context in enumerate(batch):
            ids = adapter.encode_context(context, MASK_CHAR)
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[b, : len(ids)] = 1

        # The task table is sorted by context, so this batch's terms are one slice
        lo = int(np.searchsorted(table.ctx_id, start, "left"))
        hi = int(np.searchsorted(table.ctx_id, start + len(batch), "left"))
        if lo == hi:
            continue
        local_ctx = table.ctx_id[lo:hi].astype(np.int64) - start
        key = local_ctx * max_len + table.pos[lo:hi]
        uniq, inverse = np.unique(key, return_inverse=True)
        # (context, position) is the identity of a gathered distribution: one
        # context may be read at several positions, because mask-fill masks a
        # variant's whole mutated set in a single context
        rows = torch.from_numpy((uniq // max_len).astype(np.int64)).to(device)
        cols = torch.from_numpy((uniq % max_len).astype(np.int64) + offset).to(device)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            logits = adapter.logits_at(input_ids, attention_mask, rows, cols)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        log_probs = log_probs.cpu().numpy()

        vocab = vocab_of_base[table.base[lo:hi]]
        values = log_probs[inverse, vocab] * table.sign[lo:hi]
        if not np.isfinite(values).all():
            bad = lo + int(np.flatnonzero(~np.isfinite(values))[0])
            raise FloatingPointError(
                f"{adapter.name} produced a non-finite log probability for "
                f"context {table.ctx_id[bad]}, position {table.pos[bad]}, "
                f"base {adapter.bases[table.base[bad]]}"
            )
        np.add.at(
            flat,
            table.strategy[lo:hi].astype(np.int64) * n_rows + table.row[lo:hi],
            values,
        )

    if not np.isfinite(scores[:, table.scorable]).all():
        raise FloatingPointError(
            f"{adapter.name} produced a non-finite accumulated score"
        )
    return scores
