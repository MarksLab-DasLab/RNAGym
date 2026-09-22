"""Shared AlphaFold 3 configuration readers for AF3-style prediction adapters."""

import json
from dataclasses import dataclass
from pathlib import Path

from rnagym.config import Config3D

AF3_DIR = Config3D.PREDICTION_DIR / "af3"


@dataclass(frozen=True)
class Chain:
    """One polymer chain of a benchmark target."""

    molecule: str
    chain_id: str
    sequence: str
    msa: Path | None


def _find_prepared(prepared: dict, chain_id: str) -> dict:
    """Find one chain after AF3 collapses identical sequences."""
    for item in prepared["sequences"]:
        chain = next(iter(item.values()))
        ids = chain["id"] if isinstance(chain["id"], list) else [chain["id"]]
        if chain_id in ids:
            return chain
    raise KeyError(f"AF3-prepared input is missing chain {chain_id}")


def _prepared_file(name: str, kind: str) -> Path:
    """Return the cached AF3 inputs holding generated partner-chain MSAs."""
    return Config3D.CACHE_DIR / "af3_inputs" / kind / f"{name}.json"


def _write_text(path: Path, contents: str) -> None:
    """Write one input only when its contents changed."""
    if not path.is_file() or path.read_text() != contents:
        path.write_text(contents)


def config_dependencies(config: Path, kind: str) -> tuple[Path, ...]:
    """Return every source file used to prepare one target."""
    raw = json.loads(config.read_text())
    dependencies = [config]
    dependencies.extend(
        Path(chain["unpairedMsaPath"])
        for item in raw["sequences"]
        for chain in item.values()
        if "unpairedMsaPath" in chain
    )
    if kind == "multimers":
        dependencies.append(_prepared_file(raw["name"].lower(), kind))
    return tuple(dependencies)


def config_name(config: Path) -> str:
    """Return the lowercase target name of one configuration."""
    return json.loads(config.read_text())["name"].lower()


def read_chains(config: Path, kind: str, input_dir: Path) -> list[Chain]:
    """Read one target's chains, writing out partner-chain MSAs as needed.

    Parameters
    ----------
    config : Path
        One AlphaFold 3 configuration written by the AF3 adapter.
    kind : str
        Target type, either ``monomers`` or ``multimers``.
    input_dir : Path
        Directory receiving protein alignments extracted from AF3's cache.

    Returns
    -------
    list[Chain]
        Every polymer chain, with a resolved MSA path where one exists.
    """
    raw = json.loads(config.read_text())
    prepared = None
    if kind == "multimers":
        path = _prepared_file(raw["name"].lower(), kind)
        if not path.is_file():
            raise FileNotFoundError(f"Missing AF3-prepared input: {path}")
        prepared = json.loads(path.read_text())

    input_dir.mkdir(parents=True, exist_ok=True)
    chains = []
    for item in raw["sequences"]:
        molecule, chain = next(iter(item.items()))
        chain_id = chain["id"]
        msa = None
        if "unpairedMsaPath" in chain:
            msa = Path(chain["unpairedMsaPath"])
        elif molecule in {"protein", "rna"} and prepared is not None:
            # AF3 generates partner-chain alignments inline during its data
            # pipeline, for RNA partners that are not benchmark targets as well
            alignment = _find_prepared(prepared, chain_id).get("unpairedMsa")
            if alignment:
                msa = input_dir / f"{chain_id}.a3m"
                _write_text(msa, alignment)
        chains.append(Chain(molecule, chain_id, chain["sequence"], msa))
    return chains


def sequence_length(config: Path) -> int:
    """Return the total polymer length in one configuration."""
    return sum(
        len(next(iter(item.values()))["sequence"])
        for item in json.loads(config.read_text())["sequences"]
    )


def targets(kind: str) -> list[Path]:
    """Load AF3 configurations for one target type from longest to shortest."""
    return sorted(
        (AF3_DIR / kind).glob("*/config.json"),
        key=lambda config: (-sequence_length(config), str(config)),
    )
