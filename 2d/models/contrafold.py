"""CONTRAfold model adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

from models.utils import run, sum_pair_probabilities


def _predict(
    sequence: str,
    executable: str | Path = "contrafold",
    parameters: Path | None = None,
) -> list[float]:
    with tempfile.TemporaryDirectory(prefix="rnagym-contrafold-") as tmpdir:
        input_file = Path(tmpdir) / "sequence.bpseq"
        output_file = Path(tmpdir) / "probabilities.txt"
        # BPSEQ columns are position, nucleotide, and pairing partner (-1 means
        # unknown/predict)
        input_file.write_text(
            "".join(
                f"{i}\t{base}\t-1\n"
                for i, base in enumerate(sequence.replace("T", "U"), start=1)
            )
        )
        command = f"{executable} predict {input_file}"
        if parameters:
            command += f" --params {parameters}"
        command += f" --posteriors 0.0000000001 {output_file}"
        run(command)

        # Example: 1 A 8:0.25 12:0.60
        # Nucleotide 1 pairs with 8 at probability 0.25 or 12 at probability 0.60
        pairs = []
        for line in output_file.read_text().splitlines():
            fields = line.split()
            position = int(fields[0]) - 1
            for pair in fields[2:]:
                partner, probability = pair.split(":")
                pairs.append((position, int(partner) - 1, float(probability)))

        return sum_pair_probabilities(len(sequence), pairs)


def predict(sequence: str) -> list[float]:
    """Return the probability that each nucleotide is paired."""
    return _predict(sequence)
