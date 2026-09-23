#!/usr/bin/env bash

set -euo pipefail

mkdir -p .rg_jobs
if [[ -n "$(squeue -h -u "$USER" -n rg-mmseqs-db,rg-mmseqs-search,rg-mmseqs-finish)" ]]; then
	echo "MMseqs2 generation is already queued"
	exit
fi

python -m rnagym.tasks.mmseqs inventory
database=$(sbatch --parsable --job-name=rg-mmseqs-db \
	--time=1-00:00:00 --cpus-per-task=8 --mem=64G \
	--partition=short --array=0-1 -- tasks/mmseqs.slurm prepare)
search=$(sbatch --parsable --job-name=rg-mmseqs-search \
	--dependency="aftercorr:$database" --kill-on-invalid-dep=yes \
	--time=7-00:00:00 --cpus-per-task=32 --mem=200G \
	--partition=medium --array=0-1 -- tasks/mmseqs.slurm search)
finish=$(sbatch --parsable --job-name=rg-mmseqs-finish \
	--dependency="afterok:$search" --kill-on-invalid-dep=yes \
	--time=1-00:00:00 --cpus-per-task=4 --mem=128G \
	--partition=short -- tasks/mmseqs.slurm finish)
echo "MMseqs2 databases: $database, searches: $search, alignment export: $finish"
