#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CPU_PARTITION="short"
GPU_PARTITION="gpu"

environment="$1"

case $environment in
vienna | contrafold | eternafold | rnastructure)
	shards=64
	resources=(
		--time=12:00:00
		--cpus-per-task=1
		--mem=8G
		--partition="$CPU_PARTITION"
	)
	;;
ribonanzanet | ufold)
	shards=8
	resources=(
		--time=04:00:00
		--cpus-per-task=1
		--mem=8G
		--gpus=1
		--partition="$GPU_PARTITION"
	)
	;;
*)
	echo "unknown environment: $environment" >&2
	exit 2
	;;
esac

cd "$PROJECT_ROOT"
mkdir -p .rg_predict_out

check_complete() {
	local output_dir="$PROJECT_ROOT/../data/2d/chemical_mapping/predictions/$environment/$1"
	for ((shard = 0; shard < shards; shard++)); do
		[[ -s "$output_dir/$shard.parquet" ]] || return 1
	done
}

for dataset in chemical_mapping pseudobase; do
	if check_complete "$dataset"; then
		echo "Skipping $environment $dataset: all shards complete"
		continue
	fi
	sbatch "${resources[@]}" --array="0-$((shards - 1))" \
		-- scripts/predict.slurm "$environment" "$dataset"
done
