#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${EVO2_MODEL_NAME:-evo2_40b}
prediction_name=${EVO2_PREDICTION_NAME:-$model_name}
row_id=${1:-${SLURM_ARRAY_TASK_ID:-0}}
local_path=${EVO2_LOCAL_PATH:-}

extra_args=()
if [[ -n "$local_path" ]]; then
	extra_args+=(--checkpoint "$local_path")
fi

cd "$repository"
python -m rnagym.fitness.baselines.Evo.score_evo2_single_dms \
	--rows "$row_id" \
	--output "$data_dir/model_predictions/$prediction_name" \
	--model "$model_name" \
	"${extra_args[@]}"
