#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CPU_PARTITION="short"
GPU_PARTITION="gpu"

environment="$1"
shift

case $environment in
arnie)
	resources=(
		--array=0-63
		--time=08:00:00
		--cpus-per-task=1
		--mem=8G
		--partition="$CPU_PARTITION"
	)
	;;
ribonanzanet)
	resources=(
		--array=0-7
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
sbatch "${resources[@]}" -- scripts/predict.slurm "$environment" "$@"
