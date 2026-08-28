#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# The gated checkpoint requires deepspeed and omits quantization.py. Copy that
# file from biomap-research/xtrimopglm-1b-mlm into the local checkpoint
export model_dir="path/to/local/RNAGenesis"

export reference_sheet="reference_sheet.csv"
export output_scores_dir="path/to/model_predictions/rnagenesis_4fill"
export dms_data_dir="path/to/dms/data/dir"

DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python "$script_dir/score_rnagenesis_single_dms.py" \
	--row_id "$DMS_index" \
	--ref_sheet "$reference_sheet" \
	--dms_dir_path "$dms_data_dir" \
	--output_dir_path "$output_scores_dir" \
	--model_name "$model_dir"
