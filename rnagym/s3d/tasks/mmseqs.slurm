#!/usr/bin/env bash

#SBATCH --account=marks_dsm1
#SBATCH --output=.rg_jobs/%x-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PATH="$PWD/.pixi/envs/default/bin:$PATH"
export PYTHONUNBUFFERED=1
if [[ "$1" == prepare ]]; then
	python -m rnagym.tasks.prepare_mmseqs
else
	python -m rnagym.tasks.mmseqs "$@"
fi
