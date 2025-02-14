#!/bin/bash

# script used to score both evo models. (evo-1-8k-base and evo-1.5-8k-base)
export model_name="evo-1.5-8k-base"

export reference_sheet="reference_sheet.csv"
export output_scores_dir="path/to/output/scores/dir"
export dms_data_dir="path/to/dms/data/dir"

# Get the current index from the array (0-31)
DMS_index=0

# Run the scoring script with the array task ID
python score_evo_single_dms.py \
    --row_id "$DMS_index" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --output_dir_path "$output_scores_dir" \
    --model_name "$model_name"

