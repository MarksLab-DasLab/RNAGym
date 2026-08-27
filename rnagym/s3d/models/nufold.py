"""NuFold prediction adapter."""

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
    """Run the official NuFold inference script."""
    input_dir = work_dir / "input" / sequence_id
    fasta, _ = prepare_inputs(sequence_id, input_dir, sequence_id)
    run_ipknot(fasta, input_dir / f"{sequence_id}.ipknot.ss")
    output = work_dir / "output"
    command = (
        f"{sys.executable} {Config3D.NUFOLD_DIR / 'run_nufold.py'} "
        f"--ckpt_path {Config3D.NUFOLD_PARAM_FILE} --input_fasta {fasta} "
        f"--input_dir {work_dir / 'input'} --output_dir {output} "
        "--config_preset initial_training"
    )
    run(command, cwd=Config3D.NUFOLD_DIR)
    return output / sequence_id / f"{sequence_id}_rank_1.pdb"


ADAPTER = monomer_adapter(
    "nu", "output/{sequence_id}/{sequence_id}_rank_1.pdb", _predict
)
