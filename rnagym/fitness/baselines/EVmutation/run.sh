#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)

cd "$repository"
python -m rnagym.fitness.baselines.EVmutation.compute_fitness --cpu "${SLURM_CPUS_PER_TASK:-8}"
