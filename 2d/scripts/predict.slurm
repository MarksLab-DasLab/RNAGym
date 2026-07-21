#!/usr/bin/env bash

#SBATCH --job-name=rg-predict
#SBATCH --output=.rg_predict_out/%x-%A_%a.out
#SBATCH --array=0-63
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=short

set -euo pipefail

environment="$1"
script="scripts/score_$environment.py"
shift 1

cd "$SLURM_SUBMIT_DIR"
pixi run -e "$environment" python "$script" "$@" \
	--shard "$SLURM_ARRAY_TASK_ID" \
	--num-shards "$SLURM_ARRAY_TASK_COUNT"
