"""Score fitness assays with two substitution baselines that need no trained model.

``mutation_count`` is the negative number of substituted bases. ``RNA-2Di`` sums
Riboseek's dinucleotide substitution log odds over the dinucleotides a variant
changes, relative to the wild type's own scores.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from rnagym.config import ConfigFitness
from rnagym.fitness.data import read_assay, read_reference, write_csv_atomically
from rnagym.fitness.tasks.model_registry import SCORE_COLS, resolve_source

BASES = str.maketrans("ACGTU", "01233")
# Matrix letters of the dinucleotides AA, AC, ..., UU, then of A, C, G and U
# followed by the unknown base that ends every sequence
# https://github.com/steineggerlab/riboseek/blob/ef28e4bd0d023f683d9ba40a871a99baecb2bfcf/src/commons/DinucleotideMapping.cpp#L106-L141
LETTERS = "CGLQDFRKMAPIENHS" + "TVWY"
# Riboseek v1.0.1 RNA-2Di_avg, vendored with its MIT license
MATRIX_FILE = Path(__file__).with_name("riboseek") / "dinuc.out"


def dinucleotides(codes: np.ndarray) -> np.ndarray:
    """Index each base with its successor as ``4 * a + b`` and the last as ``16 + a``.

    https://github.com/steineggerlab/riboseek/blob/ef28e4bd0d023f683d9ba40a871a99baecb2bfcf/src/commons/DinucleotideMapping.cpp#L410-L434

    Examples
    --------
    >>> dinucleotides(encode(["ACGU"]))
    array([[ 1,  6, 11, 19]], dtype=uint8)
    """
    return np.concatenate((4 * codes[:, :-1] + codes[:, 1:], 16 + codes[:, -1:]), 1)


def encode(sequences: Sequence[str]) -> np.ndarray:
    """Return base codes A, C, G, U = 0..3, reading T as U, one row per sequence."""
    text = "".join(sequences).upper().translate(BASES).encode()
    return (np.frombuffer(text, np.uint8) - ord("0")).reshape(len(sequences), -1)


def load_matrix() -> tuple[np.ndarray, float]:
    """Return RNA-2Di in ``LETTERS`` order and the scale to natural log odds.

    Scores are integers in units of 1e-4, so tied variants stay tied. Riboseek
    multiplies them by the file's lambda.
    https://github.com/steineggerlab/riboseek/blob/ef28e4bd0d023f683d9ba40a871a99baecb2bfcf/lib/mmseqs/src/commons/SubstitutionMatrix.cpp#L450-L454
    """
    lines = MATRIX_FILE.read_text().splitlines()
    scale = next(float(line.split()[-1]) for line in lines if "Lambda" in line)
    header, *rows = (line.split() for line in lines if not line.startswith("#"))
    order = [header.index(letter) for letter in LETTERS]
    values = np.array([row[1:] for row in rows], dtype=float)[np.ix_(order, order)]
    return np.rint(values * 1e4).astype(np.int64), scale / 1e4


def main(argv: Sequence[str] | None = None) -> None:
    """Score the selected reference-sheet assays with both baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="all", help="Reference rows: all, 12 or 0-8")
    rows = parser.parse_args(argv).rows
    matrix, scale = load_matrix()
    reference = read_reference(ConfigFitness.REFERENCE_FILE, rows)
    for name, wild_type in reference.select("DMS_ID", "RAW_CONSTRUCT_SEQ").iter_rows():
        assay = read_assay(ConfigFitness.ASSAY_DIR / f"{name}.csv")
        codes, wt_codes = encode(assay["sequence"].to_list()), encode([wild_type])
        mt, (wt,) = dinucleotides(codes), dinucleotides(wt_codes)
        # A changed dinucleotide wt -> mt adds S(wt, mt) - S(wt, wt)
        changed, cols = np.nonzero(mt != wt)
        terms = matrix[wt[cols], mt[changed, cols]] - matrix[wt, wt][cols]
        totals = np.bincount(changed, weights=terms, minlength=len(codes))
        scores = {
            "RNA-2Di": scale * totals,
            "mutation_count": -(codes != wt_codes).sum(1),
        }
        for model, values in scores.items():
            folder, column = resolve_source(SCORE_COLS, model)
            write_csv_atomically(
                assay.with_columns(pl.Series(column, values)),
                ConfigFitness.PREDICTION_DIR / folder / f"{name}.csv",
            )
        print(f"Scored {name}: {assay.height} variants", flush=True)


if __name__ == "__main__":
    main()
