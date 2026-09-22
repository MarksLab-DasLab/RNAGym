"""Protenix prediction adapter."""

import json
import os
import sys
from pathlib import Path

from rnagym.config import Config3D
from rnagym.s3d.models.inputs import config_name, read_chains
from rnagym.s3d.models.utils import config_adapter, run

_ENTITIES = {"dna": "dnaSequence", "protein": "proteinChain", "rna": "rnaSequence"}


def _prepare_root() -> None:
    """Point Protenix at a writable root holding the pinned checkpoint."""
    root = Config3D.PROTENIX_ROOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoint"
    target = Config3D.PROTENIX_PARAM_FILE.parent
    if not (checkpoint.is_symlink() and checkpoint.resolve() == target.resolve()):
        # Shards prepare this concurrently, so swap the link in atomically
        temporary = checkpoint.with_name(f".checkpoint.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target)
        os.replace(temporary, checkpoint)
    os.environ["PROTENIX_ROOT_DIR"] = str(root)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(root / "torch_extensions")
    # CUDA_HOME stays at the environment root so nvcc finds its own nvvm
    # tools, and is assigned rather than defaulted so a toolkit inherited
    # from the shell cannot outrank the CUDA this environment pins
    os.environ["CUDA_HOME"] = sys.prefix
    include = Path(sys.prefix) / "targets" / "x86_64-linux" / "include"
    if (include / "cuda_runtime_api.h").is_file():
        cpath = os.environ.get("CPATH")
        os.environ["CPATH"] = f"{include}:{cpath}" if cpath else str(include)
        prepend = os.environ.get("NVCC_PREPEND_FLAGS", "")
        os.environ["NVCC_PREPEND_FLAGS"] = f"{prepend} -I{include}".strip()


def setup() -> None:
    """Warm every shared artifact before array workers run concurrently.

    Protenix downloads its reference data with a bare existence check and no
    locking, and compiles a fused CUDA kernel on first import, so parallel
    workers would race on both.
    """
    _prepare_root()
    from configs.configs_data import data_configs
    from protenix.web_service.dependency_url import URL
    from runner.inference import download_from_url

    for name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "pdb_cluster_file",
        "obsolete_release_data_csv",
    ):
        destination = Path(data_configs[name])
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {name} to {destination}", flush=True)
        # Protenix writes straight to the final path, so an interrupted
        # download would be mistaken for a complete one on the next run
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        download_from_url(URL[name], str(temporary), check_weight=False)
        temporary.replace(destination)

    print("Building the fused layer norm extension", flush=True)
    import protenix.model.layer_norm  # noqa: F401


def _best_sample(output_dir: Path) -> Path:
    """Return the highest-ranked Protenix sample.

    Protenix writes samples ordered by ranking score, so sample 0 is its own
    top choice. Fall back to the first sample when that name is absent.
    """
    candidates = sorted(output_dir.rglob("*.cif"))
    if not candidates:
        raise FileNotFoundError(f"Protenix wrote no structure to {output_dir}")
    ranked = [path for path in candidates if path.stem.endswith("sample_0")]
    return ranked[0] if ranked else candidates[0]


def _write_input(config: Path, kind: str, work_dir: Path) -> Path:
    """Write one Protenix input JSON from an AlphaFold 3 configuration."""
    input_dir = work_dir / "input"
    entities: dict[tuple[str, str], dict] = {}
    sequences = []
    for chain in read_chains(config, kind, input_dir):
        # Protenix derives entity IDs from list position, so copies of one
        # sequence share an entry, and name their chains explicitly rather
        # than accepting the A, B, C... it would otherwise assign
        entity = entities.get((chain.molecule, chain.sequence))
        if entity is None:
            entity = {"sequence": chain.sequence, "count": 0, "id": []}
            if chain.msa is not None:
                entity["unpairedMsaPath"] = str(chain.msa.resolve())
            entities[chain.molecule, chain.sequence] = entity
            sequences.append({_ENTITIES[chain.molecule]: entity})
        entity["count"] += 1
        entity["id"].append(chain.chain_id)
    document = [
        {
            "name": config_name(config),
            "modelSeeds": [0],
            "assembly_id": "1",
            "sequences": sequences,
        }
    ]
    path = input_dir / "protenix.json"
    path.write_text(json.dumps(document, indent=2))
    return path


def _predict(config: Path, kind: str, work_dir: Path) -> Path:
    """Run the official Protenix inference for one target."""
    _prepare_root()
    input_json = _write_input(config, kind, work_dir)
    output_dir = work_dir / "output"
    command = (
        f"protenix pred -i {input_json} -o {output_dir} -s 0 "
        f"-n {Config3D.PROTENIX_MODEL} --use_default_params true "
        "--use_rna_msa true"
    )
    run(command)
    return _best_sample(output_dir)


ADAPTER = config_adapter("protenix", "prediction.cif", _predict, setup=setup)
