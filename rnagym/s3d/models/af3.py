"""AlphaFold 3 prediction adapter."""

import json
import shutil
import sys
from pathlib import Path

import gemmi
import polars as pl

from rnagym.config import Config3D
from rnagym.s3d.models.utils import (
    Adapter,
    is_fresh,
    prepare_msa,
    run,
    valid_structure,
)

_AF3_TYPES = {
    gemmi.PolymerType.Dna: "dna",
    # The benchmark's hybrid chains are RNA-dominant
    gemmi.PolymerType.DnaRnaHybrid: "rna",
    gemmi.PolymerType.PeptideL: "protein",
    gemmi.PolymerType.Rna: "rna",
}
_PREDICTION_DIR = Config3D.PREDICTION_DIR / "af3"


def _af3_chain_id(asym_id: str) -> str:
    """Convert a Gemmi assembly subchain ID to an AF3 chain ID."""
    base, separator, copy = asym_id.partition("-")
    if not separator:
        return base
    # AF3 accepts uppercase letters only, so encode A-2 as ADA
    return f"{base}D{base * (int(copy) - 1)}"


def _cache_inputs(configs: list[Path], output_dir: Path, cache_dir: Path) -> int:
    """Preserve prepared inputs so interrupted predictions skip database searches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = 0
    for config in configs:
        name = json.loads(config.read_text())["name"].lower()
        destination = cache_dir / f"{name}.json"
        dependencies = _config_dependencies(config)
        if is_fresh(destination, dependencies):
            continue
        destination.unlink(missing_ok=True)
        source = output_dir / name / f"{name}_data.json"
        if not source.is_file():
            source = _prediction_path(config).parent / f"{name}_data.json"
        if is_fresh(source, dependencies):
            temporary = destination.with_suffix(".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            cached += 1
    return cached


def _config_dependencies(config: Path) -> tuple[Path, ...]:
    """Return the configuration and explicit MSA inputs for one AF3 target."""
    sequences = json.loads(config.read_text())["sequences"]
    msas = (
        Path(chain["unpairedMsaPath"])
        for item in sequences
        for chain in item.values()
        if "unpairedMsaPath" in chain
    )
    return (config, *msas)


def _chain_data(
    block: gemmi.cif.Block, structure: gemmi.Structure
) -> dict[str, tuple[str, str]]:
    """Return each polymer chain's AF3 molecule type and full sequence."""
    molecules = {}
    for entity in structure.entities:
        molecule = _AF3_TYPES.get(entity.polymer_type)
        if molecule is not None:
            molecules.update(dict.fromkeys(entity.subchains, molecule))
    sequences = {asym_id: "" for asym_id in molecules}
    table = block.find_mmcif_category("_pdbx_poly_seq_scheme")
    for asym_id, residue in zip(
        table.find_column("asym_id"), table.find_column("mon_id")
    ):
        if asym_id not in sequences:
            continue
        unknown = "X" if molecules[asym_id] == "protein" else "N"
        code = gemmi.find_tabulated_residue(residue).one_letter_code.upper()
        sequences[asym_id] += code if len(code) == 1 and code.isalpha() else unknown
    return {
        asym_id: (
            molecule,
            sequences[asym_id].replace("T", "U")
            if molecule == "rna"
            else sequences[asym_id],
        )
        for asym_id, molecule in molecules.items()
    }


def _collect(configs: list[Path], output_dir: Path) -> int:
    """Move completed predictions into their permanent directories."""
    completed = 0
    for config in configs:
        source = _prediction_path(config, output_dir)
        if not _output_complete(source):
            continue
        destination = _prediction_path(config)
        if destination.parent.exists():
            shutil.rmtree(destination.parent)
        shutil.move(source.parent, destination.parent)
        (config.parent / "SUCCESS").touch()
        completed += 1
    return completed


def _monomer_configs(targets: pl.DataFrame) -> int:
    """Write one AF3 configuration per unique monomer sequence."""
    targets = (
        targets.filter(pl.col("type") == "monomer")
        .unique("sequence_id")
        .sort("sequence_id")
    )
    for (sequence_id,) in targets.select("sequence_id").iter_rows():
        run_dir = _PREDICTION_DIR / "monomers" / sequence_id
        msa, sequence = prepare_msa(sequence_id, run_dir / "sequence.a3m", unknown="X")
        chain = {
            "id": "A",
            "sequence": sequence,
            "unpairedMsaPath": str(msa),
        }
        _write_config(run_dir / "config.json", sequence_id, [{"rna": chain}])
    return targets.height


def _multimer_configs(targets: pl.DataFrame) -> int:
    """Write one AF3 configuration per multimeric PDB assembly."""
    target_groups: dict[str, dict[str, str]] = {}
    multimer_targets = targets.filter(pl.col("type") == "multimer").select(
        "PDB ID", "Asym. Chain ID", "sequence_id"
    )
    for pdb_id, asym_id, sequence_id in multimer_targets.iter_rows():
        target_groups.setdefault(pdb_id.lower(), {})[asym_id] = sequence_id

    for pdb_id, rna_targets in sorted(target_groups.items()):
        run_dir = _PREDICTION_DIR / "multimers" / pdb_id
        document = gemmi.cif.read(
            str(Config3D.CACHE_DIR / pdb_id / f"{pdb_id}-assembly.cif")
        )
        block = document.sole_block()
        structure = gemmi.make_structure_from_block(block)
        chain_data = _chain_data(block, structure)
        sequences = []
        for subchain in structure[0].subchains():
            asym_id = subchain.subchain_id()
            source_id = asym_id.partition("-")[0]
            if source_id not in chain_data:
                continue
            molecule, sequence = chain_data[source_id]
            chain = {"id": _af3_chain_id(asym_id), "sequence": sequence}
            if molecule == "rna" and source_id in rna_targets:
                msa, chain["sequence"] = prepare_msa(
                    rna_targets[source_id],
                    run_dir / f"sequence_{source_id}.a3m",
                    unknown="X",
                )
                chain["unpairedMsaPath"] = str(msa)
            sequences.append({molecule: chain})
        _write_config(run_dir / "config.json", pdb_id, sequences)
    return len(target_groups)


def _output_complete(prediction: Path) -> bool:
    """Check AF3's primary and final output files."""
    return (
        valid_structure(prediction)
        and (prediction.parent / "ranking_scores.csv").is_file()
    )


def _prediction_path(config: Path, root: Path | None = None) -> Path:
    """Return the primary structure output for one configuration."""
    name = json.loads(config.read_text())["name"].lower()
    return (root or config.parent) / name / f"{name}_model.cif"


def _recover() -> None:
    """Collect interrupted work before launching a new shard layout."""
    for kind in ("monomers", "multimers"):
        configs = list(_PREDICTION_DIR.glob(f"{kind}/*/config.json"))
        task_root = _PREDICTION_DIR / ".tasks" / kind
        input_cache = Config3D.CACHE_DIR / "af3_inputs" / kind
        completed = cached = 0
        for output_dir in task_root.glob("*/output"):
            completed += _collect(configs, output_dir)
            cached += _cache_inputs(configs, output_dir, input_cache)
        cached += _cache_inputs(configs, _PREDICTION_DIR / kind, input_cache)
        if task_root.exists():
            shutil.rmtree(task_root)
        print(f"Recovered {completed} {kind} predictions and {cached} prepared inputs")


def _sequence_length(config: Path) -> int:
    """Return the total polymer length in one configuration."""
    sequences = json.loads(config.read_text())["sequences"]
    return sum(len(next(iter(chain.values()))["sequence"]) for chain in sequences)


def _write_config(path: Path, name: str, sequences: list[dict]) -> None:
    """Write one AF3 input JSON."""
    config = {
        "name": name,
        "modelSeeds": [0],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(config, indent=2)
    if path.is_file() and path.read_text() == contents:
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(contents)
    temporary.replace(path)


def complete(config: Path) -> bool:
    """Check that one prediction completed successfully."""
    return is_fresh(
        config.parent / "SUCCESS", _config_dependencies(config)
    ) and _output_complete(_prediction_path(config))


def predict(configs: list[Path], kind: str, shard: int) -> None:
    """Run one AF3 shard as a batch."""
    task_dir = _PREDICTION_DIR / ".tasks" / kind / str(shard)
    input_cache = Config3D.CACHE_DIR / "af3_inputs" / kind
    cache_dir = Config3D.CACHE_DIR / "af3_compilation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if task_dir.exists():
        completed = _collect(configs, task_dir / "output")
        cached = _cache_inputs(configs, task_dir / "output", input_cache)
        print(f"Recovered {completed} predictions and {cached} prepared inputs")
        shutil.rmtree(task_dir)

    input_dir = task_dir / "input"
    output_dir = task_dir / "output"
    input_dir.mkdir(parents=True)
    for index, config in enumerate(configs):
        name = json.loads(config.read_text())["name"].lower()
        cached = input_cache / f"{name}.json"
        source = cached if is_fresh(cached, _config_dependencies(config)) else config
        (input_dir / f"{index}.json").symlink_to(source)

    command = (
        f"{sys.executable} {Config3D.AF3_DIR / 'run_alphafold.py'} "
        f"--input_dir {input_dir} --db_dir {Config3D.AF3_DATABASE_DIR} "
        f"--model_dir {Config3D.AF3_PARAM_DIR} --output_dir {output_dir} "
        f"--jax_compilation_cache_dir {cache_dir} --jackhmmer_n_cpu 2 "
        "--nhmmer_n_cpu 2"
    )
    result = run(command, check=False)
    completed = _collect(configs, output_dir)
    cached = _cache_inputs(configs, output_dir, input_cache)
    shutil.rmtree(task_dir)
    print(f"Completed {completed}/{len(configs)} assigned predictions")
    print(f"Cached {cached} prepared inputs")
    result.check_returncode()


def setup() -> None:
    """Prepare AF3 configurations and recover interrupted output."""
    targets = pl.read_parquet(
        Config3D.TARGET_FILE,
        columns=["PDB ID", "Asym. Chain ID", "type", "sequence_id"],
    )
    monomers = _monomer_configs(targets)
    multimers = _multimer_configs(targets)
    print(f"Prepared {monomers} monomer and {multimers} multimer configurations")
    _recover()


def targets(kind: str) -> list[Path]:
    """Load one target type from longest to shortest."""
    return sorted(
        _PREDICTION_DIR.glob(f"{kind}/*/config.json"),
        key=lambda config: (-_sequence_length(config), str(config)),
    )


ADAPTER = Adapter(
    complete=complete,
    kinds=("monomers", "multimers"),
    predict=predict,
    setup=setup,
    targets=targets,
)
