#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
checkpoint_root="$script_dir/../../.pixi/model-weights"
fitness_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness
export PYTHONPATH="$fitness_dir/.pixi/model-sources/RiNALMo${PYTHONPATH:+:$PYTHONPATH}"

row_id=${1:-${SLURM_ARRAY_TASK_ID:-0}}

cd "$repository"
python -m rnagym.fitness.baselines.RiNALMo.score_rinalmo_single_dms \
	--rows "$row_id" \
	--output "$data_dir/model_predictions/rinalmo_4fill" \
	--checkpoint "$checkpoint_root/rinalmo/rinalmo_giga_pretrained.pt"
