"""Score fitness assays with ViennaRNA ensemble folding and a 2D Potts model."""

import argparse
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import partial
from multiprocessing import Pool
from string import ascii_lowercase

import numpy as np
import polars as pl
import RNA

from rnagym.config import ConfigFitness
from rnagym.fitness.baselines.substitution.compute_fitness import check_sequences
from rnagym.fitness.data import read_assay, read_reference, write_csv_atomically
from rnagym.fitness.tasks.model_registry import SCORE_COLS, resolve_source

A3M_CLEAN = str.maketrans("T", "U", ". \t" + ascii_lowercase)
N_SAMPLES = 1000
PAIRS_16 = tuple(a + b for a in "ACGU" for b in "ACGU")
VALID_PAIRS = frozenset(("AU", "UA", "GC", "CG", "GU", "UG"))


def main(argv: Sequence[str] | None = None) -> None:
    """Score selected reference-sheet assays with ViennaRNA and ViennaRNA (MSA)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="all", help="Reference rows: all, 12 or 0-8")
    parser.add_argument("--cpu", type=int, default=8)
    args = parser.parse_args(argv)

    for row in read_reference(ConfigFitness.REFERENCE_FILE, args.rows).iter_rows(
        named=True
    ):
        name = row["DMS_ID"]
        wt = (row["RAW_CONSTRUCT_SEQ"] or "").upper().replace("T", "U")
        msa = _load_msa(name, wt, row.get("ALIGNMENT_QUERY_SEQ"))
        if msa is None:
            continue
        assay = read_assay(ConfigFitness.ASSAY_DIR / f"{name}.csv")
        check_sequences(name, wt, assay["sequence"])
        seqs = assay["sequence"].str.to_uppercase().str.replace_all("T", "U").to_list()
        for model, values in score_rna(wt, msa, seqs, args.cpu).items():
            folder, col = resolve_source(SCORE_COLS, model)
            write_csv_atomically(
                assay.with_columns(pl.Series(col, values)),
                ConfigFitness.PREDICTION_DIR / folder / f"{name}.csv",
            )
        print(f"Scored {name}: {assay.height} variants", flush=True)


def potts_scores(
    wt: str, msa: Sequence[str], sequences: Sequence[str]
) -> dict[str, float]:
    """Score sequences with a 2D Potts model on the WT MFE fold."""
    # Partition positions into WT MFE base pairs and unpaired sites
    bp = _pairs(str(RNA.fold(wt)[0]))
    unpaired = [i for i in range(len(wt)) if all(i not in p for p in bp)]

    # Clipped Laplace log-odds for site fields h and couplings J (gaps default to WT)
    filled = [
        "".join(wt[i] if b == "-" else b for i, b in enumerate(r))
        for r in dict.fromkeys(msa)
        if r != wt
    ]
    h = {i: _log_odds(Counter(r[i] for r in filled), wt[i], "ACGU") for i in unpaired}
    J = {  # J: base-pair couplings
        (i, j): _log_odds(Counter(r[i] + r[j] for r in filled), wt[i] + wt[j], PAIRS_16)
        for i, j in bp
    }

    # Sum unpaired site fields and base-pair couplings per sequence
    return {
        s: sum(h[i][s[i]] for i in unpaired) + sum(J[i, j][s[i] + s[j]] for i, j in bp)
        for s in sequences
    }


def score_rna(
    wt: str, msa: Sequence[str], sequences: Sequence[str], cpu: int
) -> dict[str, np.ndarray]:
    """Score variants by WT ensemble folding alone and with MSA Potts terms."""
    RNA.init_rand(1)
    wt_fc = RNA.fold_compound(wt, RNA.md(uniq_ML=1))
    wt_fc.pf()
    ensemble = [
        (db, _pairs(db), n) for db, n in Counter(wt_fc.pbacktrack(N_SAMPLES)).items()
    ]

    wt_lp = thermo_logp(wt, ensemble)
    unique = list(dict.fromkeys(sequences))
    with Pool(max(1, min(cpu, len(unique)))) as pool:
        lps = pool.map(partial(thermo_logp, ensemble=ensemble), unique)
    thermo = {s: min(0.0, lp - wt_lp) for s, lp in zip(unique, lps, strict=True)}
    potts = potts_scores(wt, msa, unique)
    return {
        "ViennaRNA": np.array([thermo[s] for s in sequences], dtype=float),
        "ViennaRNA_MSA": np.array(
            [thermo[s] + potts[s] for s in sequences], dtype=float
        ),
    }


def thermo_logp(
    seq: str, ensemble: Sequence[tuple[str, list[tuple[int, int]], int]]
) -> float:
    """Log Boltzmann probability of ``seq``'s mean energy on the WT ensemble."""
    fc = RNA.fold_compound(seq)
    _, g_ens = fc.pf()
    mean_e = sum(
        n * fc.eval_structure(_open_mismatches(seq, db, bp)) for db, bp, n in ensemble
    ) / sum(n for _, _, n in ensemble)
    return float((g_ens - mean_e) / (fc.exp_params.kT / 1000.0))  # kT is in cal/mol


def _load_msa(name: str, wt: str, raw_query: str | None) -> list[str] | None:
    """Load an assay's A3M alignment, stripping insertions and padding masked ends."""
    masked = (raw_query or "").upper().replace("T", "U")
    stem = f"{name}_masked" if "N" in masked else name
    path = ConfigFitness.MSA_DIR / "by_assay" / f"{stem}.a3m"
    if "N" not in masked and not path.is_file():
        return None
    mask = masked if "N" in masked else wt
    pad_l, pad_r = len(mask) - len(mask.lstrip("N")), len(mask) - len(mask.rstrip("N"))
    rows = [
        "-" * pad_l + "".join(b.splitlines()[1:]).translate(A3M_CLEAN) + "-" * pad_r
        for b in ("\n" + path.read_text()).split("\n>")[1:]
    ]
    unmasked = slice(pad_l, len(wt) - pad_r)
    if (
        len(mask) != len(wt)
        or not mask[unmasked]
        or "N" in mask[unmasked]
        or not rows
        or rows[0][unmasked] != wt[unmasked]
        or any(len(r) != len(wt) for r in rows)
    ):
        raise ValueError(f"Malformed A3M alignment: {path}")
    return [wt, *rows[1:]]


def _log_odds(
    counts: Counter[str], wt_key: str, alphabet: Iterable[str]
) -> dict[str, float]:
    """Clipped Laplace-smoothed log count ratio relative to ``wt_key``."""
    denom = counts[wt_key] + 1
    return {k: min(0.0, math.log((counts[k] + 1) / denom)) for k in alphabet}


def _open_mismatches(seq: str, db: str, bp: Sequence[tuple[int, int]]) -> str:
    """Open base pairs in ``db`` that cannot pair canonically on ``seq``."""
    bad = {k for i, j in bp if seq[i] + seq[j] not in VALID_PAIRS for k in (i, j)}
    return "".join("." if k in bad else c for k, c in enumerate(db)) if bad else db


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


if __name__ == "__main__":
    main()
