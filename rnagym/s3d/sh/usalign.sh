#!/usr/bin/env bash

set -euo pipefail

mkdir -p .rg_jobs
if squeue -h -u "$USER" -n rg-usalign | grep -q .; then
	echo "US-align is already queued"
	exit
fi
python -m rnagym.s3d.tasks.usalign prepare
sbatch --job-name=rg-usalign --time=24:00:00 --cpus-per-task=1 \
	--mem=16G --partition=short --array=0-63 \
	-- tasks/task.slurm rnagym.s3d.tasks.usalign
