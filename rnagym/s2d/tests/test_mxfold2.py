"""Test the MXFold2 model adapter."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..models.mxfold2 import predict

SEQUENCE = "GGGGAAAACCCC"

with tempfile.TemporaryDirectory() as tmpdir:
    fasta = Path(tmpdir) / "sequence.fa"
    fasta.write_text(f">sequence\n{SEQUENCE}\n")
    result = subprocess.run(
        ["mxfold2", "predict", "--bpp", tmpdir, fasta],
        check=True,
        capture_output=True,
        text=True,
    )
    official_bpp = np.loadtxt(Path(tmpdir) / "sequence.bpp")

official = result.stdout.splitlines()[2].split()[0]
prediction = predict(SEQUENCE)
np.testing.assert_allclose(
    prediction["probabilities"], official_bpp.sum(axis=0)[1:], atol=1e-4
)
assert prediction["structures"] == [{"method": "mfe", "dot_bracket": official}]
