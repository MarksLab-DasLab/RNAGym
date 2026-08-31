#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

ref_sheet="$script_dir/../../reference_sheet_final.csv"
dms_dir_path="path/to/dms/data/dir"
output_dir_path="path/to/output/scores/dir"
model_name="InstaDeepAI/NTv3_650M_pre"
row_id=${SLURM_ARRAY_TASK_ID:-0}

python "$script_dir/compute_fitness.py" \
	--row_id "$row_id" \
	--ref_sheet "$ref_sheet" \
	--dms_dir_path "$dms_dir_path" \
	--output_dir_path "$output_dir_path" \
	--model_name "$model_name"
