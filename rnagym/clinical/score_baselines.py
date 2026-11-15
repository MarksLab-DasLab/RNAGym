"""Score staged clinical SNV rows with simple baselines that need no pretraining.

``RNA-2Di`` sums Riboseek's dinucleotide substitution log odds over the
dinucleotides a substitution changes, relative to the wild type's own scores,
on the full staged sequence.

``ViennaRNA`` samples 1,000 structures from the wild-type McCaskill partition
function on a 201-nt window centred on the variant (``windows.crop_bounds``;
shorter transcripts such as mitochondrial tRNAs are passed whole) and computes
``min(0, log p_alt - log p_ref)`` on that ensemble, matching the single-sequence
thermodynamic fitness baseline.

Non-SNV rows are left unscored (``contained = False``, ``score = NaN``).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from rnagym.clinical import common
from rnagym.clinical.models import BASELINES, VIENNA_CAP
from rnagym.config import ConfigClinical
from rnagym.fitness.baselines.substitution.compute_fitness import (
    MATRIX_FILE,
    NUCLEOTIDES,
    dinucleotides,
    encode,
    load_matrix,
)

N_SAMPLES = 1000
VALID_PAIRS = frozenset(("AU", "UA", "GC", "CG", "GU", "UG"))
STACK = ("numpy", "polars")

# One row: (cohort, variant_index, ref_allele, alt_allele, seq_ref, seq_alt)
Row = tuple[str, int, str, str, str, str]


def _pairs(db: str) -> list[tuple[int, int]]:
    """Return zero-based ``(i, j)`` base pairs from dot-bracket notation."""
    stack: list[int] = []
    out: list[tuple[int, int]] = []
    for j, c in enumerate(db):
        if c == "(":
            stack.append(j)
        elif c == ")":
            out.append((stack.pop(), j))
    return out


def _open_mismatches(seq: str, db: str, bp: Sequence[tuple[int, int]]) -> str:
    """Open base pairs in ``db`` that cannot pair canonically on ``seq``."""
    bad = {k for i, j in bp if seq[i] + seq[j] not in VALID_PAIRS for k in (i, j)}
    return "".join("." if k in bad else c for k, c in enumerate(db)) if bad else db


def thermo_logp(
    seq: str, ensemble: Sequence[tuple[str, list[tuple[int, int]], int]]
) -> float:
    """Log Boltzmann probability of ``seq``'s mean energy on the WT ensemble."""
    import RNA

    fc = RNA.fold_compound(seq)
    _, g_ens = fc.pf()
    mean_e = sum(
        n * fc.eval_structure(_open_mismatches(seq, db, bp)) for db, bp, n in ensemble
    ) / sum(n for _, _, n in ensemble)
    return float((g_ens - mean_e) / (fc.exp_params.kT / 1000.0))


def score_wt_ensemble(wt: str, alts: Sequence[str]) -> dict[str, float]:
    """Score unique alternate windows on 1,000 WT Boltzmann structures."""
    import RNA

    RNA.init_rand(1)
    wt_fc = RNA.fold_compound(wt, RNA.md(uniq_ML=1))
    _, wt_g_ens = wt_fc.pf()
    ensemble = [
        (db, _pairs(db), n) for db, n in Counter(wt_fc.pbacktrack(N_SAMPLES)).items()
    ]
    wt_mean_e = sum(
        n * wt_fc.eval_structure(_open_mismatches(wt, db, bp)) for db, bp, n in ensemble
    ) / sum(n for _, _, n in ensemble)
    wt_lp = float((wt_g_ens - wt_mean_e) / (wt_fc.exp_params.kT / 1000.0))
    return {alt: min(0.0, thermo_logp(alt, ensemble) - wt_lp) for alt in alts}


def score_rna_2di_rows(rows: Sequence[Row]) -> tuple[pl.DataFrame, int]:
    """Score SNV rows with Riboseek's RNA-2Di log-odds matrix on the full sequence."""
    matrix, scale = load_matrix()
    n = len(rows)
    ctx_start = np.zeros(n, dtype=np.int64)
    ref_len = np.empty(n, dtype=np.int64)
    alt_len = np.empty(n, dtype=np.int64)
    contained = np.zeros(n, dtype=bool)
    scores = np.full(n, np.nan, dtype=np.float64)

    buckets: dict[int, tuple[list[int], list[str], list[str]]] = {}
    for r, (cohort, vi, ref, alt, seq_ref, seq_alt) in enumerate(rows):
        ref_len[r] = len(seq_ref)
        alt_len[r] = len(seq_alt)
        if cohort != "snv":
            continue
        if (
            len(seq_ref) != len(seq_alt)
            or len(ref) != 1
            or len(alt) != 1
            or not 0 <= vi < len(seq_ref)
        ):
            raise ValueError(f"row {r} marked snv has invalid alleles at {vi}")
        s = max(0, vi - 1)
        e = min(len(seq_ref), vi + 2)
        sub_ref, sub_alt = seq_ref[s:e], seq_alt[s:e]
        pos = vi - s
        if (
            re.fullmatch(NUCLEOTIDES, sub_ref) is None
            or re.fullmatch(NUCLEOTIDES, sub_alt) is None
            or sub_ref[:pos] != sub_alt[:pos]
            or sub_ref[pos + 1 :] != sub_alt[pos + 1 :]
            or sub_ref[pos].upper().replace("T", "U") != ref.upper().replace("T", "U")
            or sub_alt[pos].upper().replace("T", "U") != alt.upper().replace("T", "U")
        ):
            raise ValueError(f"row {r} local window does not match {ref}>{alt}")
        contained[r] = True
        idx, refs, alts = buckets.setdefault(len(sub_ref), ([], [], []))
        idx.append(r)
        refs.append(sub_ref)
        alts.append(sub_alt)

    for idx, refs, alts in buckets.values():
        wt = dinucleotides(encode(refs))
        mt = dinucleotides(encode(alts))
        changed, cols = np.nonzero(mt != wt)
        terms = (
            matrix[wt[changed, cols], mt[changed, cols]]
            - matrix[wt[changed, cols], wt[changed, cols]]
        )
        totals = np.bincount(changed, weights=terms, minlength=len(refs))
        scores[np.asarray(idx, dtype=np.int64)] = scale * totals

    frame = pl.DataFrame(
        {
            "ctx_start": ctx_start,
            "ref_len": ref_len,
            "alt_len": alt_len,
            "contained": contained,
            "score": scores,
        }
    )
    return frame, int(contained.sum())


def score_vienna_rows(
    rows: Sequence[Row], cap: int = VIENNA_CAP, cpu: int = 1
) -> tuple[pl.DataFrame, int]:
    """Score SNV rows by ViennaRNA ensemble folding on ``cap``-nt crops."""
    n = len(rows)
    ctx_start = np.empty(n, dtype=np.int64)
    ref_len = np.empty(n, dtype=np.int64)
    alt_len = np.empty(n, dtype=np.int64)
    contained = np.zeros(n, dtype=bool)
    scores = np.full(n, np.nan, dtype=np.float64)

    by_wt: dict[str, dict[str, None]] = {}
    row_pairs: list[tuple[int, str, str]] = []
    for r, (cohort, vi, ref, alt, seq_ref, seq_alt) in enumerate(rows):
        seq_ref_u = seq_ref.upper().replace("T", "U")
        seq_alt_u = seq_alt.upper().replace("T", "U")
        ref_u = ref.upper().replace("T", "U")
        alt_u = alt.upper().replace("T", "U")
        crop = common.crop_pair(seq_ref_u, seq_alt_u, vi, ref_u, alt_u, cap)
        ctx_start[r] = crop.start
        ref_len[r] = len(crop.ref)
        alt_len[r] = len(crop.alt)
        if cohort != "snv" or not crop.contained:
            continue
        if (
            re.fullmatch(NUCLEOTIDES, crop.ref) is None
            or re.fullmatch(NUCLEOTIDES, crop.alt) is None
        ):
            raise ValueError(f"row {r} cropped window has non-ACGTU nucleotides")
        contained[r] = True
        row_pairs.append((r, crop.ref, crop.alt))
        by_wt.setdefault(crop.ref, {})[crop.alt] = None

    groups = [(wt, list(alts)) for wt, alts in by_wt.items()]
    if groups:
        workers = max(1, min(cpu, len(groups)))
        if workers == 1:
            evaluated = [score_wt_ensemble(wt, alts) for wt, alts in groups]
        else:
            chunk = max(1, len(groups) // (workers * 8))
            with Pool(workers) as pool:
                evaluated = pool.starmap(score_wt_ensemble, groups, chunksize=chunk)
        lookup = {wt: res for (wt, _), res in zip(groups, evaluated, strict=True)}
        for r, wt, alt in row_pairs:
            scores[r] = lookup[wt][alt]

    frame = pl.DataFrame(
        {
            "ctx_start": ctx_start,
            "ref_len": ref_len,
            "alt_len": alt_len,
            "contained": contained,
            "score": scores,
        }
    )
    return frame, len(groups)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    default_cpu = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--names", nargs="+", required=True, help="Staged files, e.g. mt_rna"
    )
    parser.add_argument("--model", required=True, choices=sorted(BASELINES))
    parser.add_argument("--staged", type=Path, default=ConfigClinical.STAGED_DIR)
    parser.add_argument("--out", type=Path, default=ConfigClinical.SCORE_DIR)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--row-ids-dir",
        type=Path,
        help="Score only the row ids in <dir>/<name>.txt",
    )
    parser.add_argument("--cap", type=int, default=VIENNA_CAP)
    parser.add_argument("--cpu", type=int, default=default_cpu)
    return parser.parse_args(argv)


def score_file(args: argparse.Namespace, name: str, guard: common.SourceGuard) -> None:
    """Score one shard of one staged file and write it with its manifest."""
    started = time.time()
    row_ids = args.row_ids_dir / f"{name}.txt" if args.row_ids_dir else None
    staged_digest = common.staged_sha256(args.staged, name)
    frame = common.load_rows(args.staged, name, args.shard, args.num_shards, row_ids)
    columns = ("cohort", "variant_index", "ref_allele_tx", "alt_allele_tx")
    rows: list[Row] = list(frame.select(*columns, "seq_ref", "seq_alt").iter_rows())

    vienna_version: str | None = None
    checkpoint: dict[str, Any] | None = None
    cap: int | None = None
    n_samples: int | None = None
    if args.model == "RNA-2Di":
        checkpoint = common.checkpoint_identity(MATRIX_FILE, None)
        scored, n_contexts = score_rna_2di_rows(rows)
    else:
        import RNA

        vienna_version = str(RNA.__version__)
        cap = args.cap
        n_samples = N_SAMPLES
        scored, n_contexts = score_vienna_rows(rows, args.cap, args.cpu)

    result = frame.select("row_id", "cohort").hstack(scored)
    bad = result.filter(
        (pl.col("cohort") == "snv")
        & (~pl.col("contained") | ~pl.col("score").is_finite())
        | (pl.col("cohort") != "snv")
        & (pl.col("contained") | pl.col("score").is_not_nan())
    )
    if not bad.is_empty():
        raise FloatingPointError(
            f"{len(bad)} rows with invalid baseline state, e.g. {bad.row(0)}"
        )

    spec = {
        "scorer": "baseline",
        "model": args.model,
        "checkpoint": checkpoint,
        "stack": common.stack_versions(STACK),
        "viennarna": vienna_version,
        "cap": cap,
        "n_samples": n_samples,
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
        "snv_rows": int((result["cohort"] == "snv").sum()),
        "contexts": n_contexts,
        "cpu": args.cpu,
        "seconds_score": round(time.time() - started, 1),
    }
    out = args.out / name / args.model
    path = common.write_shard(out, args.shard, args.num_shards, result, manifest)
    print(
        f"{args.model} {name} shard {args.shard}/{args.num_shards}: {len(result)} rows, "
        f"{n_contexts} contexts, {manifest['seconds_score']} s -> {path}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Score one shard of each named file with one baseline."""
    args = parse_args(argv)
    guard = common.SourceGuard()
    for name in args.names:
        score_file(args, name, guard)


if __name__ == "__main__":
    main()
