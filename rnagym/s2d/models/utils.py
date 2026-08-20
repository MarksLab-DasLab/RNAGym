"""Shared model adapter types and utilities."""

from __future__ import annotations

import os
import subprocess
import warnings
from collections.abc import Iterable, Sequence
from contextlib import redirect_stdout
from io import StringIO
from os import PathLike
from pathlib import Path
from shlex import split
from string import ascii_letters
from typing import TypedDict

import numpy as np

# Arnie requires a package variable even though its BPP decoders use none
os.environ.setdefault("NUPACKHOME", "/tmp")
from arnie.pk_predictors import _hungarian, _threshknot  # noqa: E402

PAIRS = dict(zip("([{<", ")]}>"))


class Structure(TypedDict):
    """Decoded structure output."""

    method: str
    dot_bracket: str | None


class Prediction(TypedDict):
    """Model prediction output."""

    probabilities: list[float]
    structures: list[Structure]


def run(
    command: str | Sequence[str | PathLike[str]], **kwargs: object
) -> subprocess.CompletedProcess:
    """Run a command and raise its stderr on failure."""
    if isinstance(command, str):
        command = split(command)

    stdout = kwargs.pop("stdout", subprocess.DEVNULL)
    result = subprocess.run(
        command,
        stdout=stdout,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result


def warn_out_of_range(probabilities: np.ndarray) -> None:
    """Warn when paired probabilities fall outside [0, 1]."""
    if not np.all((0 <= probabilities) & (probabilities <= 1)):
        warnings.warn(
            "Pair probabilities outside [0, 1]",
            RuntimeWarning,
            stacklevel=2,
        )


def read_dot_bracket(path: str | PathLike[str]) -> str:
    """Read a dot-bracket structure from the final line of a file."""
    return Path(path).read_text().splitlines()[-1]


def sum_pair_probabilities(
    length: int, pairs: Iterable[tuple[int, int, float]]
) -> list[float]:
    """Sum a sparse probability matrix and return pair probabilities."""
    probabilities = np.zeros(length)
    for i, j, probability in pairs:
        probabilities[i] += probability
        probabilities[j] += probability
    warn_out_of_range(probabilities)
    return np.clip(probabilities, 0, 1).tolist()


def pair_probability_matrix(
    length: int, pairs: Iterable[tuple[int, int, float]]
) -> np.ndarray:
    """Expand sparse pairs into a symmetric probability matrix."""
    probabilities = np.zeros((length, length))
    for i, j, probability in pairs:
        probabilities[i, j] = probabilities[j, i] = probability
    return probabilities


def decode_pair_probabilities(probabilities: np.ndarray) -> list[Structure]:
    """Decode pair probabilities with Hungarian and ThreshKnot."""
    # https://github.com/WaymentSteeleLab/arnie/blob/660de8139bd2198bbe115adadd5bc5f12183f9f4/src/arnie/pk_predictors.py#L72-L104
    # Match RibonanzaNet's official Hungarian parameters
    # https://github.com/DasLab/rnet-inference/blob/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/src/rnet_2d.py#L110
    hungarian, _ = _hungarian(probabilities.copy(), theta=0.5, min_len_helix=1)
    with redirect_stdout(StringIO()):
        threshknot, _ = _threshknot(probabilities.copy())
    return [
        {"method": "hungarian", "dot_bracket": hungarian},
        {"method": "threshknot", "dot_bracket": threshknot},
    ]


def parse_pairs(structure: str) -> set[tuple[int, int]]:
    """Parse base pairs from extended dot-bracket notation."""
    stacks = {opener: [] for opener in PAIRS}
    openers_by_closer = {closer: opener for opener, closer in PAIRS.items()}
    pairs = set()
    for position, symbol in enumerate(structure):
        if symbol == ".":
            continue
        if symbol in stacks:
            stacks[symbol].append(position)
        elif symbol in openers_by_closer:
            opener = openers_by_closer[symbol]
            if not stacks[opener]:
                raise ValueError(f"Unmatched {symbol} in {structure}")
            pairs.add((stacks[opener].pop(), position))
        elif symbol in ascii_letters:
            # Arnie uses the first case encountered as the opener
            stacks[symbol] = [position]
            openers_by_closer[symbol.swapcase()] = symbol
        else:
            raise ValueError(f"Unknown structure symbol {symbol}")
    if any(stacks.values()):
        raise ValueError(f"Unmatched opener in {structure}")
    return pairs


def pairs_to_dot_bracket(length: int, pairs: Iterable[tuple[int, int]]) -> str:
    """Convert sparse zero-based pairs to extended dot-bracket notation."""
    pairs = [tuple(sorted(pair)) for pair in pairs]
    positions = [position for pair in pairs for position in pair]
    if any(
        len(pair) != 2 or not 0 <= pair[0] < pair[1] < length for pair in pairs
    ) or len(positions) != len(set(positions)):
        raise ValueError("Pairs are not a valid secondary structure")

    brackets = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
    brackets.extend(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))
    structure = ["."] * length
    levels: list[list[tuple[int, int]]] = []

    for left, right in sorted(pairs):
        level = None
        for candidate, level_pairs in enumerate(levels):
            if not any(
                left < i < right < j or i < left < j < right for i, j in level_pairs
            ):
                level = candidate
                break

        if level is None:
            level = len(levels)
            levels.append([])
        if level >= len(brackets):
            raise ValueError("Structure requires more dot-bracket levels")
        levels[level].append((left, right))
        structure[left], structure[right] = brackets[level]

    return "".join(structure)


def dot_bracket(contact: np.ndarray) -> str:
    """Convert contacts to dot-bracket, using letter pairs beyond ()[]{}<>."""
    contact = contact.astype(bool)
    contact = contact | contact.T
    if np.any(np.diag(contact)) or np.any(contact.sum(axis=0) > 1):
        raise ValueError("Contacts are not a valid secondary structure")
    return pairs_to_dot_bracket(len(contact), zip(*np.where(np.triu(contact, k=1))))
