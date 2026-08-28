#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# RiNALMo giga-v1 requires the official package, checkpoint and a CUDA device
# for its stored flash-attention modules
export checkpoint_path="path/to/rinalmo_giga_pretrained.pt"

export reference_sheet="reference_sheet.csv"
export output_scores_dir="path/to/model_predictions/rinalmo_4fill"
export dms_data_dir="path/to/dms/data/dir"

DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python "$script_dir/score_rinalmo_single_dms.py" \
	--row_id "$DMS_index" \
	--ref_sheet "$reference_sheet" \
	--dms_dir_path "$dms_data_dir" \
	--output_dir_path "$output_scores_dir" \
	--checkpoint_path "$checkpoint_path"
