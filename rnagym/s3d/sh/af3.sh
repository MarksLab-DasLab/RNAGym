#!/usr/bin/env bash

set -euo pipefail

pixi install -e af3
pixi run -e af3 build
pixi run python -m rnagym.s3d.tasks.prepare_af3
pixi run python -m rnagym.s3d.tasks.af3 recover
mkdir -p .rg_jobs
sbatch --job-name=rg-af3-monomer --time=12:00:00 --cpus-per-task=8 \
	--mem=64G --partition=gpu --gpus=1 --array=0-63%8 \
	-- tasks/af3.slurm monomers
sbatch --job-name=rg-af3-multimer --time=24:00:00 --cpus-per-task=8 \
	--mem=220G --partition=gpu --gpus=1 --array=0-63%8 \
	-- tasks/af3.slurm multimers
