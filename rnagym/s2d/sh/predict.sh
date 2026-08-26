#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACCOUNT="marks_dsm1"
CPU_PARTITION="short"
GPU_PARTITIONS="gpu,gpu_advanced"

environment="$1"

case $environment in
vienna | contrafold | eternafold | rnastructure | mxfold2)
	shards=64
	resources=(
		--time=12:00:00
		--cpus-per-task=1
		--mem=8G
		--partition="$CPU_PARTITION"
	)
	;;
ribonanzanet | rinalmo | ufold | rna-fm)
	shards=8
	gpu=l40s
	memory=8G
	[[ $environment == ribonanzanet || $environment == rinalmo ]] && gpu=h100
	[[ $environment == rinalmo ]] && memory=12G
	resources=(
		--time=08:00:00
		--cpus-per-task=1
		--mem="$memory"
		--gpus="$gpu:1"
		--partition="$GPU_PARTITIONS"
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
	local output_dir="$PROJECT_ROOT/../../data/2d/predictions/$environment/$1"
	for ((shard = 0; shard < shards; shard++)); do
		[[ -s "$output_dir/$shard.parquet" ]] || return 1
	done
}

for dataset in mapping 2d; do
	if check_complete "$dataset"; then
		echo "Skipping $environment $dataset: all shards complete"
		continue
	fi
	sbatch "${resources[@]}" --account="$ACCOUNT" --array="0-$((shards - 1))%32" \
		-- tasks/predict.slurm "$environment" "$dataset"
done
