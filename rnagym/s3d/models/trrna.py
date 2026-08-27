"""trRosettaRNA prediction adapter."""

import sys
from pathlib import Path

from rnagym.config import Config3D
from rnagym.s3d.models.utils import (
    monomer_adapter,
    prepare_inputs,
    run,
    run_ipknot,
)


def _predict(sequence_id: str, work_dir: Path) -> Path:
    """Run the official trRosettaRNA prediction and folding pipeline."""
    fasta, msa = prepare_inputs(sequence_id, work_dir, unknown="A")
    structure = work_dir / "sequence.dbn"
    geometries = work_dir / "sequence.npz"
    output = work_dir / "model_1.pdb"
    run_ipknot(fasta, structure)
    command = (
        f"{sys.executable} {Config3D.TRRNA_DIR / 'predict.py'} -i {msa} "
        f"-o {geometries} -ss {structure} -ss_fmt dot_bracket "
        f"-mdir {Config3D.TRRNA_DIR / 'params/model_1'} -gpu 0"
    )
    run(command, cwd=Config3D.TRRNA_DIR)
    command = (
        f"{sys.executable} {Config3D.TRRNA_DIR / 'fold.py'} -npz {geometries} "
        f"-fa {fasta} -out {output}"
    )
    run(command, cwd=Config3D.TRRNA_DIR)
    return output


ADAPTER = monomer_adapter("trRNA", "model_1.pdb", _predict)
