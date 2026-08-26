#!/usr/bin/env bash

#SBATCH --job-name=rg-predict
#SBATCH --output=.rg_predict_out/%x-%A_%a.out

set -euo pipefail

environment="$1"
dataset="$2"

cd "$SLURM_SUBMIT_DIR"
# The submitting task installs once, so array workers must not reinstall the environment
pixi run --as-is -e "$environment" python -m rnagym.s2d.tasks.predict "$environment" "$dataset" \
	--shard "$SLURM_ARRAY_TASK_ID" \
	--num-shards "$SLURM_ARRAY_TASK_COUNT"
