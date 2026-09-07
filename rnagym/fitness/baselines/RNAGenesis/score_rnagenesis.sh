#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
checkpoint_root="$script_dir/../../.pixi/model-weights"
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

row_id=${1:-${SLURM_ARRAY_TASK_ID:-0}}

cd "$repository"
python -m rnagym.fitness.baselines.RNAGenesis.score_rnagenesis_single_dms \
	--rows "$row_id" \
	--output "$data_dir/model_predictions/rnagenesis_4fill" \
	--model "$checkpoint_root/rnagenesis"
