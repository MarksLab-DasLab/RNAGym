"""RNAstructure model adapter."""

import tempfile
from pathlib import Path

import numpy as np

from .utils import (
    Prediction,
    decode_pair_probabilities,
    dot_bracket,
    pair_probability_matrix,
    read_dot_bracket,
    run,
    sum_pair_probabilities,
)


def read_ct(path: Path) -> str:
    """Read an RNAstructure CT file as extended dot bracket."""
    lines = path.read_text().splitlines()[1:]
    contacts = np.zeros((len(lines), len(lines)), dtype=bool)
    for line in lines:
        fields = line.split()
        i, j = int(fields[0]) - 1, int(fields[4]) - 1
        if j > i:
            contacts[i, j] = True
    return dot_bracket(contacts)


def predict(sequence: str) -> Prediction:
    """Return paired probabilities and decoded structures."""
    with tempfile.TemporaryDirectory(prefix="rnagym-rnastructure-") as tmpdir:
        sequence_file = Path(tmpdir) / "sequence.txt"
        partition_file = Path(tmpdir) / "partition.pfs"
        probability_file = Path(tmpdir) / "probabilities.txt"
        mfe_file = Path(tmpdir) / "mfe.dbn"
        mea_ct_file = Path(tmpdir) / "mea.ct"
        mea_file = Path(tmpdir) / "mea.dbn"
        probknot_file = Path(tmpdir) / "probknot.ct"
        sequence_file.write_text(sequence.replace("T", "U") + "\n")
        run(f"partition {sequence_file} {partition_file} -T 310")
        run(f"ProbabilityPlot {partition_file} {probability_file} -t -min 0.0000000001")
        run(f"Fold {sequence_file} {mfe_file} --MFE --bracket -T 310")
        run(f"MaxExpect {partition_file} {mea_ct_file} --structures 1")
        # MaxExpect writes no CT file when it predicts no pairs
        if mea_ct_file.is_file():
            run(f"ct2dot {mea_ct_file} 1 {mea_file}")
            mea = read_dot_bracket(mea_file)
        else:
            mea = "." * len(sequence)
        run(f"ProbKnot {partition_file} {probknot_file}")
        probknot = read_ct(probknot_file)

        # Example: 1 10 2.9577 means nucleotides 1 and 10 pair with probability 10^-2.9577
        pairs = []
        for line in probability_file.read_text().splitlines()[2:]:
            i, j, negative_log_probability = line.split()
            pairs.append(
                (
                    int(i) - 1,
                    int(j) - 1,
                    10 ** -float(negative_log_probability),
                )
            )

        structures = [
            {"method": "mfe", "dot_bracket": read_dot_bracket(mfe_file)},
            {"method": "mea_gamma_1", "dot_bracket": mea},
            {"method": "probknot", "dot_bracket": probknot},
        ]
        structures += decode_pair_probabilities(
            pair_probability_matrix(len(sequence), pairs)
        )
        return {
            "probabilities": sum_pair_probabilities(len(sequence), pairs),
            "structures": structures,
        }
