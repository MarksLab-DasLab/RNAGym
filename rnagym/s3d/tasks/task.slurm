#!/usr/bin/env bash

#SBATCH --account=marks_dsm1
#SBATCH --output=.rg_jobs/%x-%j.out

set -euo pipefail

module="$1"
cd "$SLURM_SUBMIT_DIR"
pixi run --as-is python -m "$module" \
	"${SLURM_ARRAY_TASK_ID:-0}" "${SLURM_ARRAY_TASK_COUNT:-1}"
