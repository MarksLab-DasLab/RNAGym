#!/usr/bin/env bash

set -euo pipefail

mkdir -p .rg_jobs
if squeue -h -u "$USER" -n rg-riboseek-db | grep -q .; then
	echo "Riboseek database setup is already queued"
	exit
fi
sbatch --job-name=rg-riboseek-db --time=7-00:00:00 \
	--cpus-per-task=32 --mem=200G --partition=medium --array=0-1 \
	-- tasks/task.slurm rnagym.tasks.prepare_riboseek
