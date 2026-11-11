"""Score fitness assays with ViennaRNA ensemble folding and a 2D Potts model."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
import RNA

from rnagym.config import ConfigFitness
from rnagym.fitness.baselines.substitution.compute_fitness import check_sequences
from rnagym.fitness.data import read_assay, read_reference, write_csv_atomically
from rnagym.fitness.tasks.model_registry import SCORE_COLS, resolve_source

DINUCLEOTIDES = tuple(a + b for a in "ACGU" for b in "ACGU")
N_SAMPLES = 1000
VALID_PAIRS = frozenset(("AU", "UA", "GC", "CG", "GU", "UG"))


def load_msa(path: Path, wt: str, query: str, offset: int) -> list[str]:
    """Read an A3M alignment, strip insertions and pad masked ends to ``wt``."""
    pad_r = "-" * (len(wt) - offset - len(query))
    chunks = re.split(r"^>[^\n]*(?:\n|$)", path.read_text().lstrip(), flags=re.M)[1:]
    rows = [
        "-" * offset + re.sub(r"[a-z.\s]", "", s).upper().replace("T", "U") + pad_r
        for s in chunks
    ]
    if (
        not rows
        or rows[0][offset : offset + len(query)] != query
        or any(len(r) != len(wt) for r in rows)
    ):
        raise ValueError(f"Malformed A3M alignment: {path}")
    return [wt, *rows[1:]]


def log_odds(
    counts: Counter[str], wt_key: str, alphabet: Iterable[str]
) -> dict[str, float]:
    """Clipped Laplace-smoothed log count ratio relative to ``wt_key``."""
    denom = counts[wt_key] + 1
    return {k: min(0.0, math.log((counts[k] + 1) / denom)) for k in alphabet}


def main(argv: Sequence[str] | None = None) -> None:
    """Score selected reference-sheet assays with ViennaRNA and ViennaRNA (MSA)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="all", help="Reference rows: all, 12 or 0-8")
    parser.add_argument("--cpu", type=int, default=8)
    args = parser.parse_args(argv)
    ref = read_reference(ConfigFitness.REFERENCE_FILE, args.rows)
    if "ALIGNMENT_QUERY_SEQ" not in ref.columns:
        ref = ref.with_columns(pl.lit("").alias("ALIGNMENT_QUERY_SEQ"))
    for name, raw_wt, raw_query in ref.select(
        "DMS_ID", "RAW_CONSTRUCT_SEQ", "ALIGNMENT_QUERY_SEQ"
    ).iter_rows():
        wt = (raw_wt or "").upper().replace("T", "U")
        mask = (raw_query or "").upper().replace("T", "U")
        has_mask = "N" in mask
        offset = len(mask) - len(mask.lstrip("N")) if has_mask else 0
        query = mask.strip("N") if has_mask else wt
        a3m = (
            ConfigFitness.MSA_DIR
            / "by_assay"
            / f"{name}{'_masked' if has_mask else ''}.a3m"
        )
        if not has_mask and not a3m.is_file():
            continue
        assay = read_assay(ConfigFitness.ASSAY_DIR / f"{name}.csv")
        check_sequences(name, wt, assay["sequence"])
        msa = load_msa(a3m, wt, query, offset)
        seqs = assay["sequence"].str.to_uppercase().str.replace_all("T", "U").to_list()
        unique = list(dict.fromkeys(seqs))

        RNA.init_rand(1)
        wt_fc = RNA.fold_compound(wt, RNA.md(uniq_ML=1))
        wt_fc.pf()
        ens = [
            (db, pairs(db), n) for db, n in Counter(wt_fc.pbacktrack(N_SAMPLES)).items()
        ]
        wt_lp = thermo_logp(wt, ens)
        with Pool(max(1, min(args.cpu, len(unique)))) as pool:
            lps = pool.map(partial(thermo_logp, structures=ens), unique)
        thermo = {s: min(0.0, lp - wt_lp) for s, lp in zip(unique, lps, strict=True)}
        potts = potts_scores(wt, msa, unique)
        scores = {
            "ViennaRNA": np.array([thermo[s] for s in seqs], dtype=float),
            "ViennaRNA_MSA": np.array(
                [potts[s] + thermo[s] for s in seqs], dtype=float
            ),
        }
        for model, values in scores.items():
            folder, column = resolve_source(SCORE_COLS, model)
            write_csv_atomically(
                assay.with_columns(pl.Series(column, values)),
                ConfigFitness.PREDICTION_DIR / folder / f"{name}.csv",
            )
        print(f"Scored {name}: {assay.height} variants", flush=True)


def open_mismatches(seq: str, db: str, bp: Sequence[tuple[int, int]]) -> str:
    """Open base pairs in ``db`` that cannot pair canonically on ``seq``."""
    bad = {k for i, j in bp if seq[i] + seq[j] not in VALID_PAIRS for k in (i, j)}
    return "".join("." if k in bad else c for k, c in enumerate(db)) if bad else db


def pairs(db: str) -> list[tuple[int, int]]:
    """Return zero-based ``(i, j)`` base pairs from dot-bracket notation."""
    stack: list[int] = []
    out: list[tuple[int, int]] = []
    for j, c in enumerate(db):
        if c == "(":
            stack.append(j)
        elif c == ")":
            out.append((stack.pop(), j))
    return out


def potts_scores(
    wt: str, msa: Sequence[str], sequences: Sequence[str]
) -> dict[str, float]:
    """Score unique sequences under a clipped 2D Potts model on the WT fold."""
    ranked = sorted(
        (-sum(a == b for a, b in zip(wt, r, strict=True)), k, r)
        for k, r in enumerate(dict.fromkeys(msa))
        if r != wt and r.count("-") < 0.25 * len(wt)
    )
    fc = RNA.fold_compound(wt)
    if ranked:
        db, _ = RNA.alifold([wt, *(r for _, _, r in ranked[:9])])
        fc.hc_add_from_db(
            open_mismatches(wt, db, pairs(db)), RNA.CONSTRAINT_DB_ENFORCE_BP
        )
    bp = pairs(str(fc.mfe()[0]))
    paired = {i for p in bp for i in p}
    unpaired = [i for i in range(len(wt)) if i not in paired]
    hom = [
        "".join(wt[i] if b == "-" else b for i, b in enumerate(r))
        for r in dict.fromkeys(msa[1:])
        if r != wt
    ]
    h = {i: log_odds(Counter(r[i] for r in hom), wt[i], "ACGU") for i in unpaired}
    couplings = {
        (i, j): log_odds(
            Counter(r[i] + r[j] for r in hom), wt[i] + wt[j], DINUCLEOTIDES
        )
        for i, j in bp
    }
    return {
        s: sum(h[i][s[i]] for i in unpaired)
        + sum(couplings[i, j][s[i] + s[j]] for i, j in bp)
        for s in sequences
    }


def thermo_logp(
    seq: str, structures: Sequence[tuple[str, list[tuple[int, int]], int]]
) -> float:
    """Log Boltzmann probability of ``seq``'s mean energy over WT structures."""
    fc = RNA.fold_compound(seq)
    _, g_ens = fc.pf()
    mean_e = sum(
        n * fc.eval_structure(open_mismatches(seq, db, bp)) for db, bp, n in structures
    ) / sum(n for _, _, n in structures)
    return float((g_ens - mean_e) / (fc.exp_params.kT / 1000.0))


if __name__ == "__main__":
    main()
