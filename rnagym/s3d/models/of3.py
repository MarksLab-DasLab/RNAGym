"""OpenFold3 prediction adapter."""

import json
from pathlib import Path

from rnagym.config import Config3D
from rnagym.s3d.models.inputs import Chain, config_name, read_chains
from rnagym.s3d.models.utils import config_adapter, run

# OpenFold3 keys an alignment by its file stem and silently skips any
# name outside its configured order, so supplied MSAs take the main slot
_MAIN_ALIGNMENT = "colabfold_main"


def _best_sample(output_dir: Path, name: str) -> Path:
    """Return the OpenFold3 sample with the highest aggregated confidence."""
    candidates = sorted(output_dir.rglob("*_model.cif"))
    if not candidates:
        raise FileNotFoundError(f"OpenFold3 wrote no structure to {output_dir}")
    ranked = []
    for structure in candidates:
        confidences = structure.with_name(
            structure.name.replace("_model.cif", "_confidences_aggregated.json")
        )
        score = -1.0
        if confidences.is_file():
            summary = json.loads(confidences.read_text())
            score = summary.get("sample_ranking_score", summary.get("ptm", -1.0))
        ranked.append((score, str(structure), structure))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    print(f"Selected {ranked[0][2].name} for {name}")
    return ranked[0][2]


def _chain_msa(chain: Chain, input_dir: Path, groups: dict[str, str]) -> Path:
    """Expose one chain's alignment under a directory keyed by its sequence.

    OpenFold3 derives an alignment's representative ID from its parent
    directory name. Chains sharing a directory share one alignment, so
    distinct sequences need distinct directories, while repeated sequences
    must keep sharing the representative OpenFold3 expects to deduplicate.
    """
    group = groups.setdefault(chain.sequence, chain.chain_id)
    destination = input_dir / group / f"{_MAIN_ALIGNMENT}{chain.msa.suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = chain.msa.resolve()
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source)
    return destination


def _write_query(config: Path, kind: str, work_dir: Path) -> tuple[Path, str]:
    """Write one OpenFold3 query JSON from an AlphaFold 3 configuration."""
    input_dir = work_dir / "input"
    name = config_name(config)
    chains = []
    groups: dict[str, str] = {}
    for chain in read_chains(config, kind, input_dir):
        entry = {
            "molecule_type": chain.molecule,
            "chain_ids": chain.chain_id,
            "sequence": chain.sequence,
        }
        if chain.msa is not None:
            entry["main_msa_file_paths"] = str(_chain_msa(chain, input_dir, groups))
        # A chain without an alignment simply omits the field; `use_msas` is a
        # query-level setting and the chain model rejects unknown keys
        chains.append(entry)
    path = input_dir / "query.json"
    path.write_text(json.dumps({"queries": {name: {"chains": chains}}}, indent=2))
    return path, name


def _predict(config: Path, kind: str, work_dir: Path) -> Path:
    """Run the official OpenFold3 inference for one target."""
    query, name = _write_query(config, kind, work_dir)
    output_dir = work_dir / "output"
    command = (
        f"run_openfold predict --query-json {query} --output-dir {output_dir} "
        f"--inference-ckpt-path {Config3D.OF3_PARAM_FILE} --use-msa-server=False"
    )
    run(command)
    return _best_sample(output_dir, name)


ADAPTER = config_adapter("of3", "prediction.cif", _predict)
