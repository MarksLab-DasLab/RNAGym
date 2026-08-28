#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# RNA-FM requires the rna-fm package and RNA-FM_pretrained.pth
export checkpoint_path="path/to/RNA-FM_pretrained.pth"

export reference_sheet="reference_sheet.csv"
export output_scores_dir="path/to/model_predictions/rna_fm_4fill"
export dms_data_dir="path/to/dms/data/dir"

DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python "$script_dir/score_rna_fm_single_dms.py" \
	--row_id "$DMS_index" \
	--ref_sheet "$reference_sheet" \
	--dms_dir_path "$dms_data_dir" \
	--output_dir_path "$output_scores_dir" \
	--checkpoint_path "$checkpoint_path"
