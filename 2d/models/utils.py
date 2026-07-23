from __future__ import annotations

import subprocess
import warnings
from collections.abc import Iterable, Sequence
from os import PathLike
from shlex import split

import numpy as np


def run(command: str | Sequence[str | PathLike[str]], **kwargs: object) -> None:
    if isinstance(command, str):
        command = split(command)

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)


def warn_out_of_range(probabilities: np.ndarray) -> None:
    if not np.all((0 <= probabilities) & (probabilities <= 1)):
        warnings.warn(
            "Pair probabilities outside [0, 1]",
            RuntimeWarning,
            stacklevel=2,
        )


def sum_pair_probabilities(
    length: int, pairs: Iterable[tuple[int, int, float]]
) -> list[float]:
    """Sum a sparse probability matrix and return pair probabilities"""
    probabilities = np.zeros(length)
    for i, j, probability in pairs:
        probabilities[i] += probability
        probabilities[j] += probability
    warn_out_of_range(probabilities)
    return probabilities.tolist()


def dot_bracket(contact: np.ndarray) -> str:
    structure = ["."] * len(contact)
    for i, j in zip(*np.where(np.triu(contact, k=1))):
        structure[i] = "("
        structure[j] = ")"
    return "".join(structure)
