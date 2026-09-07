"""Load Evo weights from the revisions used for benchmark reproduction."""

from __future__ import annotations

import fcntl
import hashlib
import json
import pkgutil
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

from rnagym.config import ConfigFitness
from rnagym.fitness.tasks.model_registry import checkpoint_revision

# The 40B checkpoint is released as two byte ranges of one PyTorch file
EVO2_40B_PARTS = (
    (41126745847, "3b74fa4e6158d49265e3e270ba8869390d064358f8bf3d2af0b3e1772728f485"),
    (41126745847, "bdc4a76e0f23f8295e7061c2f0deff24f723bd916dc4cdc4d9216cac9c2d49d5"),
)


def _verify_parts(path: Path) -> None:
    """Check the merged 40B file against its pinned upstream shard hashes."""
    with path.open("rb") as handle:
        for size, expected in EVO2_40B_PARTS:
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"Truncated Evo 2 checkpoint: {path}")
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Evo 2 checkpoint checksum mismatch: {path}")
        if handle.read(1):
            raise ValueError(f"Unexpected trailing checkpoint data: {path}")


def evo2_checkpoint(model_name: str) -> str:
    """Fetch a pinned Evo 2 file, checking both shards when assembling 40B."""
    from huggingface_hub import hf_hub_download

    repo_id = f"arcinstitute/{model_name}"
    revision = checkpoint_revision(repo_id)
    filename = f"{model_name}.pt"
    if model_name != "evo2_40b":
        return hf_hub_download(repo_id, filename, revision=revision)

    destination = ConfigFitness.CHECKPOINT_DIR / "evo2" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file():
            _verify_parts(destination)
            return str(destination)
        with NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                for index in range(len(EVO2_40B_PARTS)):
                    part = hf_hub_download(
                        repo_id, f"{filename}.part{index}", revision=revision
                    )
                    with Path(part).open("rb") as source:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                output.flush()
                _verify_parts(temporary)
                temporary.chmod(0o664)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return str(destination)


def load_evo(model_name: str) -> SimpleNamespace:
    """Load pinned safetensors with Evo 0.5's config, key mapping and dtype conversion."""
    import yaml
    from evo.models import HF_MODEL_NAME_MAP
    from evo.tokenizer import CharLevelTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from stripedhyena.model import StripedHyena
    from stripedhyena.utils import dotdict

    # Evo's public constructor does not expose a checkpoint revision
    repo_id = HF_MODEL_NAME_MAP[model_name]
    revision = checkpoint_revision(repo_id)
    index_file = Path(
        hf_hub_download(
            repo_id,
            "model.safetensors.index.json",
            revision=revision,
        )
    )
    files = sorted(set(json.loads(index_file.read_text())["weight_map"].values()))
    state = {}
    for filename in files:
        path = hf_hub_download(repo_id, filename, revision=revision)
        state.update(
            (name.removeprefix("backbone."), tensor)
            for name, tensor in load_file(path).items()
        )
    if "unembed.weight" not in state:
        state["unembed.weight"] = state["embedding_layer.weight"]

    context = "131k" if model_name == "evo-1-131k-base" else "8k"
    resource = pkgutil.get_data("evo", f"configs/evo-1-{context}-base_inference.yml")
    if resource is None:
        raise FileNotFoundError(f"Evo {context} inference config is missing")
    model = StripedHyena(dotdict(yaml.safe_load(resource), Loader=yaml.FullLoader))
    model.load_state_dict(state, strict=True)
    model.to_bfloat16_except_poles_residues()
    return SimpleNamespace(model=model, tokenizer=CharLevelTokenizer(512))
