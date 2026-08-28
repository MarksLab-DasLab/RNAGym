#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# AIDO.RNA-1.6B is an encoder-only masked language model and fits on a single
# 48 GB GPU in bfloat16. It requires torch, transformers and modelgenerator
export model_name="genbio-ai/AIDO.RNA-1.6B"

export reference_sheet="reference_sheet.csv"
export output_scores_dir="path/to/model_predictions/aido_rna_4fill"
export dms_data_dir="path/to/dms/data/dir"

DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python "$script_dir/score_aido_rna_single_dms.py" \
	--row_id "$DMS_index" \
	--ref_sheet "$reference_sheet" \
	--dms_dir_path "$dms_data_dir" \
	--output_dir_path "$output_scores_dir" \
	--model_name "$model_name"
