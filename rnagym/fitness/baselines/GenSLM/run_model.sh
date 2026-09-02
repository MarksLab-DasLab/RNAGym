#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
fitness_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness
row_id=${SLURM_ARRAY_TASK_ID:-0}

cd "$repository"
python -m rnagym.fitness.baselines.GenSLM.compute_fitness \
	--reference_sheet "$data_dir/reference_sheet_final.csv" \
	--task_id "$row_id" \
	--checkpoint_dir "$fitness_dir/.pixi/model-weights/genslm" \
	--dms_directory "$data_dir/assays" \
	--output_directory "$data_dir/model_predictions/GenSLM"
