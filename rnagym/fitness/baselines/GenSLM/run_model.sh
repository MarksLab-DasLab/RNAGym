#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
row_id=${1:-${SLURM_ARRAY_TASK_ID:-0}}

cd "$repository"
python -m rnagym.fitness.baselines.GenSLM.compute_fitness \
	--rows "$row_id"
