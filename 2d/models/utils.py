from __future__ import annotations

import subprocess
import warnings
from collections.abc import Iterable, Sequence
from os import PathLike
from pathlib import Path
from shlex import split
from typing import TypedDict

import numpy as np


class Structure(TypedDict):
    method: str
    dot_bracket: str


class Prediction(TypedDict):
    probabilities: list[float]
    structures: list[Structure]


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


def read_dot_bracket(path: str | PathLike[str]) -> str:
    return Path(path).read_text().splitlines()[-1]


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
    """Convert a contact map to extended dot-bracket notation.

    Uppercase/lowercase pairs encode levels beyond ()[]{}<>.
    """
    brackets = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
    brackets.extend(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))
    structure = ["."] * len(contact)
    levels: list[list[tuple[int, int]]] = []

    for i, j in zip(*np.where(np.triu(contact, k=1))):
        level = None
        for candidate, pairs in enumerate(levels):
            if not any(
                i < left < j < right or left < i < right < j for left, right in pairs
            ):
                level = candidate
                break

        if level is None:
            level = len(levels)
            levels.append([])

        if level >= len(brackets):
            raise ValueError("Structure requires more dot-bracket levels")
        levels[level].append((i, j))
        structure[i], structure[j] = brackets[level]

    return "".join(structure)
