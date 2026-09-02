#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
fitness_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

row_id=${SLURM_ARRAY_TASK_ID:-0}

cd "$repository"
python -m rnagym.fitness.baselines.RNAGenesis.score_rnagenesis_single_dms \
	--row_id "$row_id" \
	--ref_sheet "$data_dir/reference_sheet_final.csv" \
	--dms_dir_path "$data_dir/assays" \
	--output_dir_path "$data_dir/model_predictions/rnagenesis_4fill" \
	--model_name "$fitness_dir/.pixi/model-weights/rnagenesis"
