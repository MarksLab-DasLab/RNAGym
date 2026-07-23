"""RNAstructure model adapter."""

import tempfile
from pathlib import Path

from models.utils import run, sum_pair_probabilities


def predict(sequence: str) -> list[float]:
    """Return the probability that each nucleotide is paired."""
    with tempfile.TemporaryDirectory(prefix="rnagym-rnastructure-") as tmpdir:
        sequence_file = Path(tmpdir) / "sequence.txt"
        partition_file = Path(tmpdir) / "partition.pfs"
        probability_file = Path(tmpdir) / "probabilities.txt"
        sequence_file.write_text(sequence.replace("T", "U") + "\n")
        run(f"partition {sequence_file} {partition_file} -T 310")
        run(f"ProbabilityPlot {partition_file} {probability_file} -t -min 0.0000000001")

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

        return sum_pair_probabilities(len(sequence), pairs)
