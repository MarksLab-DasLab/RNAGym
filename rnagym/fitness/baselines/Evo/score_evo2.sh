#!/usr/bin/env bash
set -euo pipefail
umask 002

checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$checkpoint_root/evo2/hub"}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${EVO2_MODEL_NAME:-evo2_40b}
prediction_name=${EVO2_PREDICTION_NAME:-$model_name}
rows=${1:-${SLURM_ARRAY_TASK_ID:-all}}
local_path=${EVO2_LOCAL_PATH:-}

extra_args=()
if [[ -n "$local_path" ]]; then
	extra_args+=(--checkpoint "$local_path")
fi

cd "$repository"
python -m rnagym.fitness.baselines.Evo.score_evo2_single_dms \
	--rows "$rows" \
	--output "$data_dir/model_predictions/$prediction_name" \
	--model "$model_name" \
	"${extra_args[@]}"
