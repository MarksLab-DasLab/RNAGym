#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
rows=${1:-${SLURM_ARRAY_TASK_ID:-all}}

cd "$repository"
python -m rnagym.fitness.baselines.GenSLM.compute_fitness \
	--rows "$rows"
