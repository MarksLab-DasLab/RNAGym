#!/usr/bin/env bash

#SBATCH --account=marks_dsm1
#SBATCH --output=.rg_jobs/%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.pixi/envs/default/bin:$PATH"

args=("$@")
if [[ "$1" == search ]]; then
	args+=("$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT")
fi
python -m rnagym.sequences "${args[@]}"
