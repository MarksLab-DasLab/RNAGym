#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)

cd "$repository"
python -m rnagym.fitness.baselines.RNAERNIE.compute_fitness --rows "${1:-${SLURM_ARRAY_TASK_ID:-all}}"
