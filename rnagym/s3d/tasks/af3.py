"""Run one shard of AlphaFold 3 predictions."""

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from rnagym.config import Config3D

PREDICTION_DIR = Config3D.PREDICTION_DIR / "af3"


def prediction_path(config: Path, root: Path | None = None) -> Path:
    """Return the primary structure output for one configuration."""
    name = json.loads(config.read_text())["name"].lower()
    return (root or config.parent) / name / f"{name}_model.cif"


def output_complete(prediction: Path) -> bool:
    """Check AF3's primary and final output files."""
    return (
        prediction.is_file()
        and prediction.stat().st_size > 100
        and (prediction.parent / "ranking_scores.csv").is_file()
    )


def is_complete(config: Path) -> bool:
    """Check that one prediction completed successfully."""
    return (config.parent / "SUCCESS").is_file() and output_complete(
        prediction_path(config)
    )


def sequence_length(config: Path) -> int:
    """Return the total polymer length in one configuration."""
    sequences = json.loads(config.read_text())["sequences"]
    return sum(len(next(iter(chain.values()))["sequence"]) for chain in sequences)


def collect(configs: list[Path], output_dir: Path) -> int:
    """Move completed predictions into their permanent directories."""
    completed = 0
    for config in configs:
        source = prediction_path(config, output_dir)
        if not output_complete(source):
            continue
        destination = prediction_path(config)
        if destination.parent.exists():
            shutil.rmtree(destination.parent)
        shutil.move(source.parent, destination.parent)
        (config.parent / "SUCCESS").touch()
        completed += 1
    return completed


def cache_inputs(configs: list[Path], output_dir: Path, cache_dir: Path) -> int:
    """Preserve prepared inputs so interrupted predictions can skip database searches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = 0
    for config in configs:
        name = json.loads(config.read_text())["name"].lower()
        source = output_dir / name / f"{name}_data.json"
        if source.is_file():
            shutil.move(source, cache_dir / f"{name}.json")
            cached += 1
    return cached


def recover() -> None:
    """Collect interrupted work before launching a new shard layout."""
    for group in ("monomers", "multimers"):
        configs = list(PREDICTION_DIR.glob(f"{group}/*/config.json"))
        task_root = PREDICTION_DIR / ".tasks" / group
        input_cache = Config3D.CACHE_DIR / "af3_inputs" / group
        completed = cached = 0
        for output_dir in task_root.glob("*/output"):
            completed += collect(configs, output_dir)
            cached += cache_inputs(configs, output_dir, input_cache)
        if task_root.exists():
            shutil.rmtree(task_root)
        print(f"Recovered {completed} {group} predictions and {cached} prepared inputs")


def main() -> None:
    """Run the assigned AlphaFold 3 configurations."""
    if sys.argv[1:] == ["recover"]:
        recover()
        return
    task_id, task_count = map(int, sys.argv[1:3])
    kind = sys.argv[3] if len(sys.argv) > 3 else "*"
    if kind not in {"*", "monomers", "multimers"}:
        raise ValueError(f"Unknown target type: {kind}")
    configs = sorted(
        PREDICTION_DIR.glob(f"{kind}/*/config.json"),
        key=lambda config: (-sequence_length(config), str(config)),
    )
    configs = configs[task_id::task_count]
    group = "all" if kind == "*" else kind
    task_dir = PREDICTION_DIR / ".tasks" / group / str(task_id)
    input_cache = Config3D.CACHE_DIR / "af3_inputs" / group
    cache_dir = Config3D.CACHE_DIR / "af3_compilation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if task_dir.exists():
        completed = collect(configs, task_dir / "output")
        cached = cache_inputs(configs, task_dir / "output", input_cache)
        print(f"Recovered {completed} predictions and {cached} prepared inputs")
        shutil.rmtree(task_dir)
    configs = [config for config in configs if not is_complete(config)]
    if not configs:
        print("All assigned predictions are complete")
        return

    input_dir = task_dir / "input"
    output_dir = task_dir / "output"
    input_dir.mkdir(parents=True)
    for index, config in enumerate(configs):
        cached = input_cache / f"{json.loads(config.read_text())['name'].lower()}.json"
        (input_dir / f"{index}.json").symlink_to(cached if cached.is_file() else config)

    command = f"""
        {shlex.quote(sys.executable)} {shlex.quote(str(Config3D.AF3_DIR / "run_alphafold.py"))}
        --input_dir {shlex.quote(str(input_dir))}
        --db_dir {shlex.quote(str(Config3D.AF3_DATABASE_DIR))}
        --model_dir {shlex.quote(str(Config3D.AF3_PARAM_DIR))}
        --output_dir {shlex.quote(str(output_dir))}
        --jax_compilation_cache_dir {shlex.quote(str(cache_dir))}
        --jackhmmer_n_cpu 2
        --nhmmer_n_cpu 2
    """
    result = subprocess.run(shlex.split(" ".join(command.split())), check=False)
    completed = collect(configs, output_dir)
    cached = cache_inputs(configs, output_dir, input_cache)
    shutil.rmtree(task_dir)
    print(f"Completed {completed}/{len(configs)} assigned predictions")
    print(f"Cached {cached} prepared inputs")
    result.check_returncode()


if __name__ == "__main__":
    main()
