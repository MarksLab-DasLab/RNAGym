#!/bin/bash

export reference_sheet="reference_sheet.csv"
export output_scores_dir="path/to/output/scores/dir"
export dms_data_dir="path/to/dms/data/dir"

# https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
export model_location="InstaDeepAI/nucleotide-transformer-2.5b-multi-species"

# Get the current index from the array (0-31)
DMS_index=0

python compute_fitness.py \
  --reference_sheet "$reference_sheet" \
  --task_id "$DMS_index" \
  --model_location "$model_location" \
  --dms_directory "$dms_data_dir" \
  --output_directory "$output_scores_dir"
