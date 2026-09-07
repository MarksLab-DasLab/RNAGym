#!/usr/bin/env bash
set -euo pipefail
umask 002

checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$checkpoint_root/aido-rna/hub"}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${AIDO_RNA_MODEL_NAME:-genbio-ai/AIDO.RNA-1.6B}
prediction_name=${AIDO_RNA_PREDICTION_NAME:-aido_rna_4fill}
row_id=${1:-${SLURM_ARRAY_TASK_ID:-0}}

cd "$repository"
python -m rnagym.fitness.baselines.AIDO_RNA.score_aido_rna_single_dms \
	--rows "$row_id" \
	--output "$data_dir/model_predictions/$prediction_name" \
	--model "$model_name"
