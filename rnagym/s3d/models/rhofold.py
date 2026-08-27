"""RhoFold prediction adapter."""

import sys
from pathlib import Path
from types import SimpleNamespace

from rnagym.config import Config3D
from rnagym.s3d.models.utils import monomer_adapter, prepare_inputs


def _predict(sequence_id: str, work_dir: Path) -> Path:
    """Run the official RhoFold inference without optional relaxation."""
    fasta, msa = prepare_inputs(sequence_id, work_dir)
    output = work_dir / "output"
    sys.path.insert(0, str(Config3D.RHOFOLD_DIR))
    import inference

    # Upstream downloads the checkpoint on every call even when it exists
    inference.snapshot_download = lambda **_: None
    inference.main(
        SimpleNamespace(
            ckpt=str(Config3D.RHOFOLD_PARAM_FILE),
            device="cuda:0",
            input_a3m=str(msa),
            input_fas=str(fasta),
            output_dir=str(output),
            relax_steps=0,
            single_seq_pred=False,
        )
    )
    return output / "unrelaxed_model.pdb"


ADAPTER = monomer_adapter("rho", "output/unrelaxed_model.pdb", _predict)
