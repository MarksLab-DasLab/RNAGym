"""Boltz-2 prediction adapter."""

from pathlib import Path

from rnagym.config import Config3D
from rnagym.s3d.models.inputs import config_name, read_chains
from rnagym.s3d.models.utils import config_adapter, run


def _write_input(config: Path, kind: str, work_dir: Path) -> tuple[Path, str]:
    """Write one Boltz FASTA input from an AlphaFold 3 configuration.

    Boltz accepts alignments for protein chains only, so RNA and DNA chains
    are folded from their sequence alone.
    """
    input_dir = work_dir / "input"
    name = config_name(config)
    records = []
    alignments: dict[str, str] = {}
    for chain in read_chains(config, kind, input_dir):
        fields = [chain.chain_id, chain.molecule]
        if chain.molecule == "protein":
            alignment = str(chain.msa.resolve()) if chain.msa else "empty"
            # Boltz rejects an input where copies of one sequence name
            # different alignments, so every copy reuses the first
            fields.append(alignments.setdefault(chain.sequence, alignment))
        records.append(f">{'|'.join(fields)}\n{chain.sequence}\n")
    path = input_dir / f"{name}.fasta"
    path.write_text("".join(records))
    return path, name


def _predict(config: Path, kind: str, work_dir: Path) -> Path:
    """Run the official Boltz-2 inference for one target."""
    fasta, name = _write_input(config, kind, work_dir)
    output_dir = work_dir / "output"
    command = (
        f"boltz predict {fasta} --out_dir {output_dir} "
        f"--cache {Config3D.BOLTZ_CACHE_DIR} --output_format mmcif "
        # Boltz seeds its sampler only when a seed is given
        "--model boltz2 --accelerator gpu --devices 1 --override --seed 0"
    )
    run(command)
    # Boltz ranks its samples, writing model 0 as the top choice
    predictions = output_dir / f"boltz_results_{name}" / "predictions" / name
    prediction = predictions / f"{name}_model_0.cif"
    if prediction.is_file():
        return prediction
    candidates = sorted(output_dir.rglob("*_model_*.cif"))
    if not candidates:
        raise FileNotFoundError(f"Boltz-2 wrote no structure to {output_dir}")
    return candidates[0]


ADAPTER = config_adapter("boltz2", "prediction.cif", _predict)
