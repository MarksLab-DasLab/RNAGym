#!/usr/bin/env bash

set -euo pipefail

work_dir="$1"
search_shards=8

mkdir -p .rg_jobs
prepare=$(sbatch --parsable --job-name=rg-cluster-prepare --time=02:00:00 \
	--cpus-per-task=16 --mem=32G --partition=short \
	-- tasks/cluster.slurm prepare "$work_dir")
search=$(sbatch --parsable --job-name=rg-cluster-search \
	--dependency="afterok:$prepare" --time=12:00:00 --cpus-per-task=32 \
	--mem=48G --partition=short --array="0-$((search_shards - 1))" \
	-- tasks/cluster.slurm search "$work_dir")
cluster=$(sbatch --parsable --job-name=rg-cluster \
	--dependency="afterok:$search" --time=04:00:00 --cpus-per-task=16 \
	--mem=64G --partition=short \
	-- tasks/cluster.slurm cluster "$work_dir")
echo "Prepare $prepare, search $search, cluster $cluster"
