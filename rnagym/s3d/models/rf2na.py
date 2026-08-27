"""RoseTTAFold2NA prediction adapter."""

import json
import shutil
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from rnagym.config import Config3D
from rnagym.s3d.models.utils import Adapter, is_fresh, run, valid_structure

_AF3_DIR = Config3D.PREDICTION_DIR / "af3"
_PREDICTION_DIR = Config3D.PREDICTION_DIR / "rf2na"


def _add_templates(specification: str) -> str:
    """Add RF2NA template files to one protein input specification."""
    if not specification.startswith("P:"):
        return specification
    msa = Path(specification.split(":", 1)[1])
    return f"{specification}:{msa.with_suffix('.hhr')}:{msa.with_suffix('.atab')}"


def _find_chain(prepared: dict, chain_id: str) -> dict:
    """Find a chain after AF3 collapses identical sequences."""
    for item in prepared["sequences"]:
        chain = next(iter(item.values()))
        ids = chain["id"] if isinstance(chain["id"], list) else [chain["id"]]
        if chain_id in ids:
            return chain
    raise KeyError(f"AF3-prepared input is missing chain {chain_id}")


def _input_dependencies(config: Path, kind: str) -> tuple[Path, ...]:
    """Return every source file used to prepare one RF2NA target."""
    raw = json.loads(config.read_text())
    dependencies = [config]
    dependencies.extend(
        Path(chain["unpairedMsaPath"])
        for item in raw["sequences"]
        for chain in item.values()
        if "unpairedMsaPath" in chain
    )
    if kind == "multimers":
        dependencies.append(
            Config3D.CACHE_DIR / "af3_inputs" / kind / f"{raw['name']}.json"
        )
    return tuple(dependencies)


def _normalize_msa(text: str) -> str:
    """Unwrap an MSA and map unsupported uppercase RNA residues to N."""
    records = []
    for line in text.splitlines():
        if line.startswith(">"):
            records.append([line, ""])
        else:
            records[-1][1] += line.strip()
    normalized = []
    for header, sequence in records:
        sequence = "".join(
            base if base in "ACGUTN-" or base.islower() else "N" for base in sequence
        )
        normalized.append(f"{header}\n{sequence}\n")
    return "".join(normalized)


def _output_complete(output_dir: Path) -> bool:
    """Check the primary RF2NA prediction files."""
    return (
        valid_structure(output_dir / "model_00.pdb")
        and (output_dir / "model_00.npz").is_file()
        and (output_dir / "model_00.npz").stat().st_size > 100
    )


def _predict(config: Path, kind: str) -> None:
    """Predict one target with the official RF2NA model."""
    target_dir = _PREDICTION_DIR / kind / config.parent.name
    output_dir = target_dir / "models"
    inputs = _write_inputs(config, target_dir / "input", kind)
    if kind == "multimers":
        inputs = [_add_templates(specification) for specification in inputs]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    command = (
        f"{sys.executable} {Config3D.RF2NA_DIR / 'network/predict.py'} "
        f"-inputs {' '.join(inputs)} -prefix {output_dir / 'model'} "
        f"-model {Config3D.RF2NA_PARAM_FILE} -db {Config3D.RF2NA_DATABASE}"
    )
    run(command, cwd=Config3D.RF2NA_DIR / "network")
    if not _output_complete(output_dir):
        raise RuntimeError(
            f"RF2NA did not produce complete output for {config.parent.name}"
        )
    (target_dir / "SUCCESS").touch()


def _prepare_templates(config: Path) -> None:
    """Prepare official RF2NA protein template hits for one multimer."""
    target_dir = _PREDICTION_DIR / "multimers" / config.parent.name
    for specification in _write_inputs(config, target_dir / "input", "multimers"):
        molecule, msa_path = specification.split(":", 1)
        if molecule != "P":
            continue
        msa = Path(msa_path)
        hhr = msa.with_suffix(".hhr")
        atab = msa.with_suffix(".atab")
        if is_fresh(hhr, (msa,)) and is_fresh(atab, (msa,)):
            continue
        command = (
            "hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu 4 "
            f"-maxmem 32 -aliw 100000 -e 100 -p 5.0 -d {Config3D.RF2NA_DATABASE} "
            f"-i {msa} -o {hhr} -atab {atab} -v 0"
        )
        run(command)


def _run_targets(configs: list[Path], action: Callable[[Path], None]) -> None:
    """Finish independent targets before reporting failures."""
    failures = []
    for config in configs:
        try:
            action(config)
        except Exception:
            traceback.print_exc()
            failures.append(config.parent.name)
    if failures:
        raise RuntimeError(f"Failed targets: {', '.join(failures)}")


def _sequence_length(config: Path) -> int:
    """Return the total polymer length in one AF3 configuration."""
    return sum(
        len(next(iter(item.values()))["sequence"])
        for item in json.loads(config.read_text())["sequences"]
    )


def _write_inputs(config: Path, input_dir: Path, kind: str) -> list[str]:
    """Write RF2NA inputs using MSAs supplied to or generated by AF3."""
    raw = json.loads(config.read_text())
    prepared = None
    if kind == "multimers":
        path = Config3D.CACHE_DIR / "af3_inputs" / kind / f"{raw['name']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing AF3-prepared input: {path}")
        prepared = json.loads(path.read_text())

    input_dir.mkdir(parents=True, exist_ok=True)
    specifications = []
    for item in raw["sequences"]:
        molecule, chain = next(iter(item.items()))
        chain_id = chain["id"]
        af3_chain = _find_chain(prepared, chain_id) if prepared else chain
        suffix = {"dna": "fa", "protein": "a3m", "rna": "afa"}[molecule]
        path = input_dir / f"{chain_id}.{suffix}"
        for stale in input_dir.glob(f"{chain_id}.*"):
            if stale.suffix in {".a3m", ".afa", ".fa"} and stale != path:
                stale.unlink()
        if molecule == "rna":
            msa_path = chain.get("unpairedMsaPath")
            msa = Path(msa_path).read_text() if msa_path else af3_chain["unpairedMsa"]
            _write_text(path, _normalize_msa(msa))
            specifications.append(f"R:{path}")
        elif molecule == "protein":
            _write_text(path, af3_chain["unpairedMsa"])
            specifications.append(f"P:{path}")
        elif molecule == "dna":
            # RF2NA uses D for unknown DNA residues
            sequence = "".join(
                base if base in "ACGT" else "D" for base in chain["sequence"].upper()
            )
            _write_text(path, f">query\n{sequence}\n")
            specifications.append(f"S:{path}")

    return specifications


def _write_text(path: Path, contents: str) -> None:
    """Write one input only when its contents changed."""
    if not path.is_file() or path.read_text() != contents:
        path.write_text(contents)


def complete(config: Path) -> bool:
    """Check that one RF2NA prediction completed."""
    kind = config.parent.parent.name
    target_dir = _PREDICTION_DIR / kind / config.parent.name
    return is_fresh(
        target_dir / "SUCCESS", _input_dependencies(config, kind)
    ) and _output_complete(target_dir / "models")


def predict(configs: list[Path], kind: str, _shard: int) -> None:
    """Predict one shard of RF2NA targets."""
    _run_targets(configs, lambda config: _predict(config, kind))


def prepare(configs: list[Path], kind: str) -> None:
    """Prepare RF2NA template inputs for one shard."""
    if kind != "multimers":
        return
    _run_targets(configs, _prepare_templates)


def targets(kind: str) -> list[Path]:
    """Load AF3 configurations from longest to shortest."""
    return sorted(
        (_AF3_DIR / kind).glob("*/config.json"),
        key=lambda config: (-_sequence_length(config), str(config)),
    )


ADAPTER = Adapter(
    complete=complete,
    kinds=("monomers", "multimers"),
    predict=predict,
    prepare=prepare,
    targets=targets,
)
