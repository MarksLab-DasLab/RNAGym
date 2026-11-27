#!/usr/bin/env bash
set -euo pipefail
umask 002

checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$checkpoint_root/structrfm/hub"}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${STRUCTRFM_MODEL_NAME:-heqin-zhu/structRFM}
prediction_name=${STRUCTRFM_PREDICTION_NAME:-structrfm_4fill}
rows=${1:-${SLURM_ARRAY_TASK_ID:-all}}

cd "$repository"
python -m rnagym.fitness.baselines.structRFM.score_structrfm_single_dms \
	--rows "$rows" \
	--output "$data_dir/model_predictions/$prediction_name" \
	--model "$model_name"
