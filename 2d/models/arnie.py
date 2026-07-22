"""Arnie model adapter."""

import os
import tempfile
from pathlib import Path

import numpy as np

# Arnie discovers "RNAstructure" but later looks up lowercase "rnastructure"
_TEMP_DIR = tempfile.TemporaryDirectory(prefix="rnagym-arnie-")
_ARNIEFILE = Path(_TEMP_DIR.name) / "arniefile.txt"
_ARNIEFILE.write_text(
    f"rnastructure: {os.environ['RNASTRUCTURE_PATH']}\nTMP: {_TEMP_DIR.name}\n"
)
os.environ["ARNIEFILE"] = str(_ARNIEFILE)
# Arnie reads ARNIEFILE when imported, so this must follow the setup above
from arnie.bpps import bpps  # noqa: E402


def predict(sequence: str, model: str) -> list[float]:
    """Return the probability that each nucleotide is paired."""
    # ViennaRNA ignores Arnie's TMP setting and writes files to the working directory
    working_dir = Path.cwd()
    try:
        os.chdir(_TEMP_DIR.name)
        pair_probabilities = bpps(sequence.replace("T", "U"), package=model)
    finally:
        os.chdir(working_dir)
    return np.clip(pair_probabilities.sum(axis=1), 0, 1).tolist()
