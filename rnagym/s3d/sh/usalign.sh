#!/usr/bin/env bash

set -euo pipefail

mkdir -p .rg_jobs
pixi run --as-is python -c \
	'from rnagym.s3d.util.analysis import prep_usalign; prep_usalign()'
sbatch --job-name=rg-usalign --time=24:00:00 --cpus-per-task=1 \
	--mem=16G --partition=short --array=0-63 \
	-- tasks/task.slurm rnagym.s3d.tasks.usalign
