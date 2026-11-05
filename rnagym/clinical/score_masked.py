"""Score staged clinical rows with a masked language model.

SNV: the masked marginal at the variant, ``log p(alt) - log p(ref)`` with the
site masked in the reference window (single-substitution ``wt-fill``).

Other variants (indels, MNVs): a local pseudo-log-likelihood. Each position
within ``local_k`` of the variant is masked in turn, in the reference window and
in the alternate window, and the score is ``mean PLL(alt) - mean PLL(ref)``.

Windows are cropped to the model's cap at a shared start centred on the variant
(``windows.crop_bounds``). Batching and logit gathering reuse the fitness
engine and adapters unchanged; contexts are bucketed by length because the
engine never mixes lengths in a batch.
"""

from __future__ import annotations

import argparse
import importlib
import time
from array import array
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from rnagym.clinical import common
from rnagym.clinical.models import MASKED
from rnagym.config import ConfigClinical
from rnagym.fitness.baselines.masked_lm.adapter import MaskedLMAdapter
from rnagym.fitness.baselines.masked_lm.engine import accumulate_scores, pad_contexts
from rnagym.fitness.baselines.masked_lm.strategies import (
    MASK_CHAR,
    TaskTable,
    validate_table,
)
from rnagym.fitness.tasks.model_registry import checkpoint_revision

STACK = (
    "torch",
    "transformers",
    "numpy",
    "polars",
    "modelgenerator",
    "rna-fm",
    "rinalmo",
    "mamba-ssm",
    "flash-attn",
)

STRATEGIES = ("site", "alt_sum", "ref_sum")
SITE, ALT_SUM, REF_SUM = range(3)

# One row: (cohort, variant_index, ref_allele, alt_allele, seq_ref, seq_alt)
Row = tuple[str, int, str, str, str, str]


class _Bucket:
    """Contexts of one length and their accumulation terms."""

    def __init__(self) -> None:
        """Start an empty bucket."""
        self.index: dict[str, int] = {}
        self.contexts: list[str] = []
        self.ctx = array("i")
        self.pos = array("i")
        self.base = array("b")
        self.row = array("i")
        self.strategy = array("b")
        self.sign = array("b")

    def add(
        self, context: str, pos: int, base: int, row: int, strategy: int, sign: int
    ) -> None:
        """Add one term, reusing the context if the bucket already holds it."""
        key = self.index.get(context)
        if key is None:
            key = self.index[context] = len(self.contexts)
            self.contexts.append(context)
        self.ctx.append(key)
        self.pos.append(pos)
        self.base.append(base)
        self.row.append(row)
        self.strategy.append(strategy)
        self.sign.append(sign)

    def table(self, n_rows: int) -> TaskTable:
        """Return the bucket as a task table, terms ordered by context."""
        order = np.argsort(np.frombuffer(self.ctx, dtype=np.int32), kind="stable")

        def take(values: array[int], dtype: type) -> npt.NDArray[Any]:
            return np.frombuffer(values, dtype=dtype)[order].copy()

        rows = take(self.row, np.int32)
        scorable = np.zeros(n_rows, dtype=bool)
        scorable[rows] = True
        table = TaskTable(
            contexts=self.contexts,
            ctx_id=take(self.ctx, np.int32),
            pos=take(self.pos, np.int32),
            base=take(self.base, np.int8),
            row=rows,
            strategy=take(self.strategy, np.int8),
            sign=take(self.sign, np.int8),
            scorable=scorable,
            strategies=STRATEGIES,
        )
        validate_table(table)
        return table


def _masked(window: str, pos: int) -> str:
    return window[:pos] + MASK_CHAR + window[pos + 1 :]


def build_buckets(
    rows: list[Row],
    bases: str,
    canonical: Callable[[str], str],
    cap: int,
    local_k: int,
) -> tuple[dict[int, _Bucket], pl.DataFrame]:
    """Turn rows into length buckets of masked contexts and per-row metadata.

    Returns the buckets and, per row, the crop and PLL counts.
    """
    base_id = {b: i for i, b in enumerate(bases)}
    buckets: dict[int, _Bucket] = {}
    names = ("ctx_start", "ref_len", "alt_len", "contained", "n_ref_pos", "n_alt_pos")
    meta: dict[str, list[Any]] = {k: [] for k in names}

    def bucket(length: int) -> _Bucket:
        return buckets.setdefault(length, _Bucket())

    for r, (cohort, vi, ref, alt, seq_ref, seq_alt) in enumerate(rows):
        seq_ref, seq_alt = canonical(seq_ref), canonical(seq_alt)
        ref, alt = canonical(ref), canonical(alt)
        crop = common.crop_pair(seq_ref, seq_alt, vi, ref, alt, cap)
        n_ref_pos = n_alt_pos = 0
        if cohort == "snv":
            context = _masked(crop.ref, crop.site)
            bucket(len(context)).add(context, crop.site, base_id[alt], r, SITE, 1)
            bucket(len(context)).add(context, crop.site, base_id[ref], r, SITE, -1)
        elif crop.contained:
            for window, allele, strategy in (
                (crop.ref, ref, REF_SUM),
                (crop.alt, alt, ALT_SUM),
            ):
                lo = max(0, crop.site - local_k)
                hi = min(len(window), crop.site + len(allele) + local_k)
                b = bucket(len(window))
                for p in range(lo, hi):
                    b.add(_masked(window, p), p, base_id[window[p]], r, strategy, 1)
                if strategy == REF_SUM:
                    n_ref_pos = hi - lo
                else:
                    n_alt_pos = hi - lo
        meta["ctx_start"].append(crop.start)
        meta["ref_len"].append(len(crop.ref))
        meta["alt_len"].append(len(crop.alt))
        meta["contained"].append(crop.contained)
        meta["n_ref_pos"].append(n_ref_pos)
        meta["n_alt_pos"].append(n_alt_pos)
    schema = {k: pl.Int64 for k in meta} | {"contained": pl.Boolean}
    return buckets, pl.DataFrame(meta, schema=schema)


def run_buckets(
    adapter: MaskedLMAdapter,
    buckets: dict[int, _Bucket],
    n_rows: int,
    max_batch_tokens: int,
) -> npt.NDArray[np.float64]:
    """Accumulate every bucket and add them up per row and strategy."""
    total = np.full((len(STRATEGIES), n_rows), np.nan)
    for length in sorted(buckets):
        table = buckets[length].table(n_rows)
        pad_contexts(table, adapter)
        scores = accumulate_scores(
            adapter,
            table,
            n_rows=n_rows,
            batch_size=adapter.default_batch_size,
            max_batch_tokens=max_batch_tokens,
            progress=False,
        )
        covered = np.isfinite(scores)
        total[covered] = np.nan_to_num(total[covered]) + scores[covered]
    return total


def finalize(
    meta: pl.DataFrame, cohorts: list[str], total: npt.NDArray[np.float64]
) -> pl.DataFrame:
    """Return per-row raw sums and the score (higher = alternate more likely)."""
    snv = np.array([c == "snv" for c in cohorts])
    n_ref = meta["n_ref_pos"].to_numpy().astype(float)
    n_alt = meta["n_alt_pos"].to_numpy().astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pll = total[ALT_SUM] / n_alt - total[REF_SUM] / n_ref
    score = np.where(
        snv, total[SITE], np.where(meta["contained"].to_numpy(), pll, np.nan)
    )
    return meta.with_columns(
        pl.Series("site_llr", np.where(snv, total[SITE], np.nan)),
        pl.Series("ref_pll_sum", np.where(snv, np.nan, total[REF_SUM])),
        pl.Series("alt_pll_sum", np.where(snv, np.nan, total[ALT_SUM])),
        pl.Series("score", score),
    )


def make_adapter(
    key: str, device: str
) -> tuple[MaskedLMAdapter, int, int, dict[str, str | Path], dict[str, Any]]:
    """Load one checkpoint's adapter and return it with its cap, budget and identity."""
    module_name, class_name, load_args, cap, max_batch_tokens = MASKED[key]
    module = importlib.import_module(module_name)
    adapter: MaskedLMAdapter = getattr(module, class_name)()
    namespace = argparse.Namespace(device=device, **load_args)
    adapter.load(namespace)
    adapter.check_alphabet()
    if adapter.context_pad_char is None and adapter.max_tokens is not None:
        if cap + adapter.n_special_tokens > adapter.max_tokens:
            raise ValueError(
                f"cap {cap} exceeds {adapter.name}'s max_tokens {adapter.max_tokens}"
            )
    checkpoint = next(iter(load_args.values()))
    local = Path(checkpoint).exists()
    revision = None if local else checkpoint_revision(str(checkpoint))
    identity = common.checkpoint_identity(checkpoint, revision)
    budget = max_batch_tokens or adapter.default_max_batch_tokens
    return adapter, cap, budget, load_args, identity


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--names", nargs="+", required=True, help="Staged files, e.g. mt_rna"
    )
    parser.add_argument("--model", required=True, choices=sorted(MASKED))
    parser.add_argument("--staged", type=Path, default=ConfigClinical.STAGED_DIR)
    parser.add_argument("--out", type=Path, default=ConfigClinical.SCORE_DIR)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--row-ids-dir",
        type=Path,
        help="Score only the row ids in <dir>/<name>.txt",
    )
    parser.add_argument("--local-k", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Score one shard of each named file with one checkpoint."""
    args = parse_args(argv)
    guard = common.SourceGuard()  # before any model code is imported
    started = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loaded = make_adapter(args.model, device)
    print(f"loaded {args.model} in {time.time() - started:.1f} s", flush=True)
    for name in args.names:
        score_file(args, name, *loaded, guard, device)


def score_file(
    args: argparse.Namespace,
    name: str,
    adapter: MaskedLMAdapter,
    cap: int,
    max_batch_tokens: int,
    load_args: dict[str, str | Path],
    identity: dict[str, Any],
    guard: common.SourceGuard,
    device: str,
) -> None:
    """Score one shard of one staged file and write it with its manifest."""
    started = time.time()
    row_ids = args.row_ids_dir / f"{name}.txt" if args.row_ids_dir else None
    staged_digest = common.staged_sha256(args.staged, name)  # before the rows are read
    frame = common.load_rows(args.staged, name, args.shard, args.num_shards, row_ids)
    columns = ("cohort", "variant_index", "ref_allele_tx", "alt_allele_tx")
    rows: list[Row] = list(frame.select(*columns, "seq_ref", "seq_alt").iter_rows())
    buckets, meta = build_buckets(
        rows, adapter.bases, adapter.canonicalize_sequence, cap, args.local_k
    )
    n_contexts = sum(len(b.contexts) for b in buckets.values())
    total = run_buckets(adapter, buckets, len(rows), max_batch_tokens)
    result = finalize(meta, frame["cohort"].to_list(), total)
    result = frame.select("row_id", "cohort").hstack(result)

    bad = result.filter(
        (pl.col("cohort") == "snv") & ~pl.col("site_llr").is_finite()
        | (pl.col("cohort") != "snv")
        & pl.col("contained")
        & ~pl.col("score").is_finite()
    )
    if not bad.is_empty():
        raise FloatingPointError(
            f"{len(bad)} rows without a finite score, e.g. {bad.row(0)}"
        )

    spec = {
        "scorer": "masked",
        "model": args.model,
        "adapter": adapter.name,
        "load_args": load_args,
        "checkpoint": identity,
        "stack": common.stack_versions(STACK),
        "cap": cap,
        "local_k": args.local_k,
        "max_batch_tokens": max_batch_tokens,
        "batch_size": adapter.default_batch_size,
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
        "contexts": n_contexts,
        "context_lengths": {
            str(k): len(b.contexts) for k, b in sorted(buckets.items())
        },
        "seconds_score": round(time.time() - started, 1),
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "torch": torch.__version__,
    }
    out = args.out / name / args.model
    path = common.write_shard(out, args.shard, args.num_shards, result, manifest)
    print(
        f"{args.model} {name} shard {args.shard}/{args.num_shards}: {len(result)} rows, "
        f"{n_contexts} contexts, {manifest['seconds_score']} s -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
