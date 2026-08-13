#!/usr/bin/env bash

set -euo pipefail

mkdir -p .rg_jobs
sbatch --job-name=rg-riboseek --time=7-00:00:00 \
	--cpus-per-task=8 --mem=200G --partition=gpu --gpus=1 --array=0-3%4 \
	-- tasks/task.slurm rnagym.tasks.riboseek
