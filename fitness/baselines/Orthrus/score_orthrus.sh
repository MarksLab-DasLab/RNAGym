#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Orthrus requires causal-conv1d, mamba-ssm and transformers remote model code
export model_name="antichronology/orthrus-mlm-6-track"

export reference_sheet="reference_sheet.csv"
export output_scores_dir="path/to/model_predictions/orthrus_4fill"
export dms_data_dir="path/to/dms/data/dir"

DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python "$script_dir/score_orthrus_single_dms.py" \
	--row_id "$DMS_index" \
	--ref_sheet "$reference_sheet" \
	--dms_dir_path "$dms_data_dir" \
	--output_dir_path "$output_scores_dir" \
	--model_name "$model_name"
