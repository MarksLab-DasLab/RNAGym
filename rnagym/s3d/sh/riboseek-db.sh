#!/usr/bin/env bash

set -euo pipefail

mkdir -p .rg_jobs
sbatch --job-name=rg-riboseek-db --time=7-00:00:00 \
	--cpus-per-task=32 --mem=200G --partition=medium --array=0-1 \
	-- tasks/task.slurm rnagym.tasks.prepare_riboseek
