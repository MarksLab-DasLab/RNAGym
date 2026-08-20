#!/usr/bin/env bash

set -euo pipefail

mkdir -p .rg_jobs
cm_workers="${1:-8}"
query_shards=2
[[ "$cm_workers" =~ ^[1-9][0-9]*$ ]] || {
	echo "CM job limit must be positive" >&2
	exit 2
}

if python -m rnagym.tasks.riboseek check 0 1; then
	echo "Riboseek is complete"
	exit
elif squeue -h -n rg-riboseek-finish | grep -q .; then
	echo "Riboseek is already queued"
	exit
fi

for ((shard = 0; shard < query_shards; shard++)); do
	if python -m rnagym.tasks.riboseek check "$shard" "$query_shards"; then
		echo "Riboseek shard $((shard + 1))/$query_shards is complete"
		continue
	fi

	search=complete
	dependency=()
	if ! python -m rnagym.tasks.riboseek ready "$shard" "$query_shards"; then
		search=$(sbatch --parsable --job-name=rg-riboseek-search \
			--time=7-00:00:00 --cpus-per-task=32 --mem=200G \
			--partition=gpu_advanced --gpus=h100:2 \
			-- tasks/riboseek.slurm search "$shard" "$query_shards")
		repair=$(sbatch --parsable --job-name=rg-riboseek-search-repair \
			--dependency="afterany:$search" --time=7-00:00:00 \
			--cpus-per-task=32 --mem=200G --partition=gpu_advanced --gpus=h100:2 \
			-- tasks/riboseek.slurm search "$shard" "$query_shards")
		dependency=(--dependency="afterok:$repair")
	fi

	cm=$(sbatch --parsable --job-name=rg-riboseek-cm "${dependency[@]}" \
		--time=7-00:00:00 --cpus-per-task=16 --mem=200G --partition=medium \
		--array="0-$((cm_workers - 1))%$cm_workers" \
		-- tasks/riboseek.slurm cmsearch "$shard" "$query_shards")
	finish=$(sbatch --parsable --job-name=rg-riboseek-finish \
		--dependency="afterany:$cm" --time=7-00:00:00 --cpus-per-task=8 \
		--mem=200G --partition=medium \
		-- tasks/riboseek.slurm finish "$shard" "$query_shards" "$cm_workers")
	echo "Shard $((shard + 1))/$query_shards: search $search, CM $cm, finish $finish"
done
