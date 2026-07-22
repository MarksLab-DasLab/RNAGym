#!/usr/bin/env bash
set -euo pipefail

CPU_PARTITION="short"
# GPU_PARTITION="gpu"
# GRES="gpu:l40s:1"

environment="$1"
shift

case $environment in
arnie)
	resources=(
		--array=0-63
		--time=12:00:00
		--cpus-per-task=1
		--mem=8G
		--partition="$CPU_PARTITION"
	)
	;;
*)
	echo "unknown environment: $environment" >&2
	exit 2
	;;
esac

mkdir -p .rg_predict_out
sbatch "${resources[@]}" -- scripts/predict.slurm "$environment" "$@"
