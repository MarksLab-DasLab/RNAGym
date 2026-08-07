"""CONTRAfold model adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .utils import (
    Prediction,
    decode_pair_probabilities,
    pair_probability_matrix,
    read_dot_bracket,
    run,
    sum_pair_probabilities,
)


def _predict(
    sequence: str,
    executable: str | Path = "contrafold",
    parameters: Path | None = None,
) -> Prediction:
    """Run a CONTRAfold-compatible executable and return its predictions."""
    with tempfile.TemporaryDirectory(prefix="rnagym-contrafold-") as tmpdir:
        input_file = Path(tmpdir) / "sequence.bpseq"
        probability_file = Path(tmpdir) / "probabilities.txt"
        viterbi_file = Path(tmpdir) / "viterbi.dbn"
        mea_file = Path(tmpdir) / "mea.dbn"
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
        # Gamma 1 gives standard maximum expected accuracy decoding
        run(
            f"{command} --gamma 1 --parens {mea_file} "
            f"--posteriors 0.0000000001 {probability_file}"
        )
        run(f"{command} --viterbi --parens {viterbi_file}")

        # Example: 1 A 8:0.25 12:0.60
        # Nucleotide 1 pairs with 8 at probability 0.25 or 12 at probability 0.60
        pairs = []
        for line in probability_file.read_text().splitlines():
            fields = line.split()
            position = int(fields[0]) - 1
            for pair in fields[2:]:
                partner, probability = pair.split(":")
                pairs.append((position, int(partner) - 1, float(probability)))

        structures = [
            {"method": "viterbi", "dot_bracket": read_dot_bracket(viterbi_file)},
            {"method": "mea_gamma_1", "dot_bracket": read_dot_bracket(mea_file)},
        ]
        structures += decode_pair_probabilities(
            pair_probability_matrix(len(sequence), pairs)
        )
        return {
            "probabilities": sum_pair_probabilities(len(sequence), pairs),
            "structures": structures,
        }


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and decoded structures."""
    return _predict(sequence)
