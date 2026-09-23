"""RoseTTAFold3 prediction adapter."""

import json
from pathlib import Path

from rnagym.config import Config3D
from rnagym.s3d.models.inputs import config_name, read_chains
from rnagym.s3d.models.utils import config_adapter, run

# AtomWorks chain types use the CIF entity_poly vocabulary
_CHAIN_TYPES = {
    "dna": "polydeoxyribonucleotide",
    "protein": "polypeptide(L)",
    "rna": "polyribonucleotide",
}


def _write_input(config: Path, kind: str, work_dir: Path) -> tuple[Path, str]:
    """Write one RoseTTAFold3 input JSON from an AlphaFold 3 configuration."""
    input_dir = work_dir / "input"
    name = config_name(config)
    components = []
    for chain in read_chains(config, kind, input_dir):
        component = {
            "seq": chain.sequence,
            "chain_id": chain.chain_id,
            # Pass the type explicitly; RF3 otherwise infers it from the alphabet
            "chain_type": _CHAIN_TYPES[chain.molecule],
        }
        if chain.msa is not None:
            component["msa_path"] = str(chain.msa.resolve())
        components.append(component)
    path = input_dir / "rf3.json"
    path.write_text(json.dumps({"name": name, "components": components}, indent=2))
    return path, name


def _predict(config: Path, kind: str, work_dir: Path) -> Path:
    """Run the official RoseTTAFold3 inference for one target."""
    input_json, name = _write_input(config, kind, work_dir)
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = (
        f"rf3 fold inputs={input_json} ckpt_path={Config3D.RF3_PARAM_FILE} "
        # A benchmark needs RF3's actual prediction for hard targets, not the
        # empty output its default early stopping would leave behind
        # RF3 falls back to the ambient random state when no seed is set
        f"out_dir={output_dir} early_stopping_plddt_threshold=0 seed=0"
    )
    run(command)
    # RF3 writes its highest-ranked fold beside the per-sample directories
    prediction = output_dir / f"{name}_model.cif"
    if prediction.is_file():
        return prediction
    candidates = sorted(output_dir.rglob("*_model.cif"))
    if not candidates:
        raise FileNotFoundError(f"RoseTTAFold3 wrote no structure to {output_dir}")
    return candidates[0]


ADAPTER = config_adapter("rf3", "prediction.cif", _predict)
