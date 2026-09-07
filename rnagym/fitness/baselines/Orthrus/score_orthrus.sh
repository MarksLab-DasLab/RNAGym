#!/usr/bin/env bash
set -euo pipefail
umask 002

checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$checkpoint_root/orthrus/hub"}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${ORTHRUS_MODEL_NAME:-antichronology/orthrus-mlm-6-track}
row_id=${1:-${SLURM_ARRAY_TASK_ID:-0}}

cd "$repository"
python -m rnagym.fitness.baselines.Orthrus.score_orthrus_single_dms \
	--rows "$row_id" \
	--output "$data_dir/model_predictions/orthrus_4fill" \
	--model "$model_name"
