#!/usr/bin/env bash

#SBATCH --account=marks_dsm1
#SBATCH --output=.rg_jobs/%x-%A_%a.out

set -euo pipefail

# Use host memory when an input exceeds the L40S memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=3.2
export PYTHONSAFEPATH=1

cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.pixi/envs/af3/bin:$PATH"
.pixi/envs/af3/bin/python -m rnagym.s3d.tasks.af3 \
	"$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" "${1:-*}"
