#!/bin/bash

# the directory containing the GenSLM checkpoints
# downloaded from https://github.com/ramanathanlab/genslm?tab=readme-ov-file#usage
# should contain a file named 2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt
export checkpoint_dir="path/to/checkpoint/dir"

export reference_sheet="reference_sheet.csv"
export output_scores_dir="path/to/output/scores/dir"
export dms_data_dir="path/to/dms/data/dir"

# Get the current index from the array (0-31)
DMS_index=0

python compute_fitness.py \
  --reference_sheet "$reference_sheet" \
  --task_id "$DMS_index" \
  --checkpoint_dir "$checkpoint_dir" \
  --dms_directory "$dms_data_dir" \
  --output_directory "$output_scores_dir"
