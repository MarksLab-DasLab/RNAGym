#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${EVO_MODEL_NAME:-evo-1.5-8k-base}
prediction_name=${EVO_PREDICTION_NAME:-evo1.5}
row_id=${1:-${SLURM_ARRAY_TASK_ID:-0}}

cd "$repository"
python -m rnagym.fitness.baselines.Evo.score_evo_single_dms \
	--rows "$row_id" \
	--output "$data_dir/model_predictions/$prediction_name" \
	--model "$model_name"
