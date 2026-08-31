#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${AIDO_RNA_MODEL_NAME:-genbio-ai/AIDO.RNA-1.6B}
prediction_name=${AIDO_RNA_PREDICTION_NAME:-aido_rna_4fill}
row_id=${SLURM_ARRAY_TASK_ID:-0}

cd "$repository"
python -m rnagym.fitness.baselines.AIDO_RNA.score_aido_rna_single_dms \
	--row_id "$row_id" \
	--ref_sheet "$data_dir/reference_sheet_final.csv" \
	--dms_dir_path "$data_dir/assays" \
	--output_dir_path "$data_dir/model_predictions/$prediction_name" \
	--model_name "$model_name"
