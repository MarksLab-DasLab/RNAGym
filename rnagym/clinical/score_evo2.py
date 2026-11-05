"""Score staged clinical rows with Evo 2: whole-window log-likelihood change.

For each row the reference and alternate windows are cropped to the model's
cap at a shared start centred on the variant. Each window's log-likelihood is
the sum over its next-token targets (positions 1..L-1; Evo 2 uses no BOS), and

    score = delta_fwd = mean LL(alt) - mean LL(ref)

with the mean over each window's own targets. This is the Evo 2 paper's
convention: variant minus reference sequence likelihood, reduce mean, forward
strand only. ``--rc`` also scores the reverse complements (``delta_rc``) as a
sensitivity analysis; it never changes ``score``.

Windows are batched only with windows of the same length, so no padding ever
enters a batch. Log-softmax runs in fp32 and sums accumulate in fp64.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from rnagym.clinical import common
from rnagym.clinical.models import EVO2
from rnagym.config import ConfigClinical, ConfigFitness
from rnagym.fitness.baselines.Evo.checkpoints import evo2_checkpoint

STACK = (
    "torch",
    "evo2",
    "vtx",
    "transformer_engine",
    "flash_attn",
    "nvidia-cublas-cu12",
    "numpy",
    "polars",
)

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

Forward = Callable[[torch.Tensor], torch.Tensor]
# One row: (variant_index, ref_allele, alt_allele, seq_ref, seq_alt)
Row = tuple[int, str, str, str, str]


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def encode(sequences: list[str]) -> npt.NDArray[np.int64]:
    """Return Evo 2's character tokens: the UTF-8 bytes, one per base."""
    length = len(sequences[0])
    joined = "".join(sequences).encode("ascii")
    tokens = np.frombuffer(joined, dtype=np.uint8).reshape(len(sequences), length)
    return tokens.astype(np.int64)


def window_log_likelihoods(
    forward: Forward, sequences: list[str], budget: int, device: str
) -> npt.NDArray[np.float64]:
    """Return the sum of next-token log probabilities of each sequence, in fp64.

    ``forward`` maps a ``(batch, length)`` long tensor to logits of shape
    ``(batch, length, vocab)``. Sequences are grouped by length so that a batch
    never holds padding; ``budget`` caps tokens per batch (at least one
    sequence per batch).
    """
    out = np.full(len(sequences), np.nan)
    by_length: dict[int, list[int]] = defaultdict(list)
    for i, seq in enumerate(sequences):
        if len(seq) < 2:
            raise ValueError("a window needs at least two bases to have a target")
        by_length[len(seq)].append(i)
    for length in sorted(by_length):
        indices = by_length[length]
        size = max(1, budget // length)
        for start in range(0, len(indices), size):
            batch = indices[start : start + size]
            ids = torch.from_numpy(encode([sequences[i] for i in batch])).to(device)
            with torch.inference_mode():
                logits = forward(ids)
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                targets = logprobs[:, :-1].gather(2, ids[:, 1:].unsqueeze(-1))
                sums = targets.squeeze(-1).double().sum(dim=1).cpu().numpy()
            out[batch] = sums
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite window log-likelihood")
    return out


def score_rows(
    forward: Forward,
    rows: list[Row],
    cap: int,
    budget: int,
    device: str,
    with_rc: bool = False,
) -> tuple[pl.DataFrame, int]:
    """Crop each row, score unique windows on both strands, return per-row deltas."""
    index: dict[str, int] = {}
    unique: list[str] = []

    def add(seq: str) -> int:
        key = index.get(seq)
        if key is None:
            key = index[seq] = len(unique)
            unique.append(seq)
        return key

    ref_ids: list[int] = []
    alt_ids: list[int] = []
    meta: dict[str, list[Any]] = {
        k: [] for k in ("ctx_start", "ref_len", "alt_len", "contained")
    }
    for vi, ref, alt, seq_ref, seq_alt in rows:
        crop = common.crop_pair(seq_ref, seq_alt, vi, ref, alt, cap)
        # A window needs two bases to have one next-token target
        scored = crop.contained and len(crop.ref) >= 2 and len(crop.alt) >= 2
        ref_ids.append(add(crop.ref) if scored else -1)
        alt_ids.append(add(crop.alt) if scored else -1)
        meta["ctx_start"].append(crop.start)
        meta["ref_len"].append(len(crop.ref))
        meta["alt_len"].append(len(crop.alt))
        meta["contained"].append(scored)

    fwd = window_log_likelihoods(forward, unique, budget, device)
    if with_rc:
        complements = [reverse_complement(s) for s in unique]
        rc = window_log_likelihoods(forward, complements, budget, device)
    else:
        rc = np.full(len(unique), np.nan)
    contained = np.array(meta["contained"], dtype=bool)
    fwd, rc = np.append(fwd, np.nan), np.append(rc, np.nan)  # index -1 reads NaN
    ref_index = np.array(ref_ids, dtype=np.int64)
    alt_index = np.array(alt_ids, dtype=np.int64)
    n_ref = np.array(meta["ref_len"], dtype=float) - 1
    n_alt = np.array(meta["alt_len"], dtype=float) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        delta_fwd = fwd[alt_index] / n_alt - fwd[ref_index] / n_ref
        delta_rc = rc[alt_index] / n_alt - rc[ref_index] / n_ref
    schema = {
        "ctx_start": pl.Int64,
        "ref_len": pl.Int64,
        "alt_len": pl.Int64,
        "contained": pl.Boolean,
    }
    frame = pl.DataFrame(meta, schema=schema).with_columns(
        pl.Series("ref_ll_fwd", fwd[ref_index]),
        pl.Series("alt_ll_fwd", fwd[alt_index]),
        pl.Series("ref_ll_rc", rc[ref_index]),
        pl.Series("alt_ll_rc", rc[alt_index]),
        pl.Series("delta_fwd", np.where(contained, delta_fwd, np.nan)),
        pl.Series("delta_rc", np.where(contained, delta_rc, np.nan)),
        pl.Series("score", np.where(contained, delta_fwd, np.nan)),
    )
    return frame, len(unique)


def checkpoint_file(key: str, cache: Path | None = None) -> str:
    """Return the pinned checkpoint file, checking the assembled 40B file once.

    ``evo2_checkpoint`` hashes both 41 GB parts of the 40B file on every call. A
    file that passed that check is reused while its identity (path, device,
    inode, size and times) is unchanged. The identity is taken before and after
    the check and recorded only if they agree, so a file moved or replaced at
    any point is checked again. The other checkpoints come from the Hub cache.
    """
    if key != "evo2_40b":
        return evo2_checkpoint(key)
    cache = cache or ConfigClinical.VERIFIED_FILE
    assembled = ConfigFitness.CHECKPOINT_DIR / "evo2" / f"{key}.pt"
    if common.known_good(assembled, key, cache):
        return str(assembled)
    before = common.file_identity(assembled) if assembled.is_file() else None
    checkpoint = evo2_checkpoint(key)
    same_file = Path(checkpoint) == assembled and assembled.is_file()
    if before is not None and same_file and common.file_identity(assembled) == before:
        common.mark_good(before, key, cache)
    return checkpoint


def load_model(key: str) -> tuple[Forward, int, int, str]:
    """Load one Evo 2 checkpoint from its pinned file and check it is FP8."""
    from evo2 import Evo2

    cap, budget = EVO2[key]
    checkpoint = checkpoint_file(key)
    model = Evo2(key, local_path=checkpoint)
    if not bool(model.model.config.get("use_fp8_input_projections", False)):
        raise RuntimeError("the constructed model does not use FP8 input projections")

    probe = "ACGTTGCAAC"
    if model.tokenizer.tokenize(probe) != encode([probe])[0].tolist():
        raise RuntimeError("Evo 2's tokenizer no longer maps bases to their bytes")

    def forward(ids: torch.Tensor) -> torch.Tensor:
        logits, _ = model.model(ids)
        return logits

    return forward, cap, budget, checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--names", nargs="+", required=True, help="Staged files, e.g. mt_rna"
    )
    parser.add_argument("--model", required=True, choices=sorted(EVO2))
    parser.add_argument("--staged", type=Path, default=ConfigClinical.STAGED_DIR)
    parser.add_argument("--out", type=Path, default=ConfigClinical.SCORE_DIR)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--row-ids-dir",
        type=Path,
        help="Score only the row ids in <dir>/<name>.txt",
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="Also score reverse complements (sensitivity)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Score one shard of each named file with one checkpoint."""
    args = parse_args(argv)
    guard = common.SourceGuard()  # before any model code is imported
    started = time.time()
    forward, cap, budget, checkpoint = load_model(args.model)
    identity = common.checkpoint_identity(checkpoint, None)
    print(f"loaded {args.model} in {time.time() - started:.1f} s", flush=True)
    for name in args.names:
        score_file(args, name, forward, cap, budget, identity, guard)


def score_file(
    args: argparse.Namespace,
    name: str,
    forward: Forward,
    cap: int,
    budget: int,
    identity: dict[str, Any],
    guard: common.SourceGuard,
) -> None:
    """Score one shard of one staged file and write it with its manifest."""
    started = time.time()
    row_ids = args.row_ids_dir / f"{name}.txt" if args.row_ids_dir else None
    staged_digest = common.staged_sha256(args.staged, name)  # before the rows are read
    frame = common.load_rows(args.staged, name, args.shard, args.num_shards, row_ids)
    columns = ("variant_index", "ref_allele_tx", "alt_allele_tx", "seq_ref", "seq_alt")
    rows: list[Row] = list(frame.select(*columns).iter_rows())
    result, n_windows = score_rows(
        forward, rows, cap, budget, "cuda:0", with_rc=args.rc
    )
    result = frame.select("row_id", "cohort").hstack(result)

    spec = {
        "scorer": "evo2_whole_window",
        "model": args.model,
        "checkpoint": identity,
        "stack": common.stack_versions(STACK),
        "cap": cap,
        "tokens_per_batch": budget,
        "reverse_complement": args.rc,
        "name": name,
        "staged_sha256": staged_digest,
        "row_ids_file": str(row_ids) if row_ids else None,
        "num_shards": args.num_shards,
        "source_sha256": guard.sha256,
    }
    guard.check()
    if common.staged_sha256(args.staged, name) != staged_digest:
        raise RuntimeError(f"staged {name} changed while this job ran; not publishing")
    manifest = {
        "spec": spec,
        "spec_hash": common.spec_hash(spec),
        "shard": args.shard,
        "rows": len(result),
        "unique_windows": n_windows,
        "seconds_score": round(time.time() - started, 1),
        "gpus": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "torch": torch.__version__,
    }
    out = args.out / name / args.model
    path = common.write_shard(out, args.shard, args.num_shards, result, manifest)
    print(
        f"{args.model} {name} shard {args.shard}/{args.num_shards}: {len(result)} rows, "
        f"{n_windows} windows, {manifest['seconds_score']} s -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
