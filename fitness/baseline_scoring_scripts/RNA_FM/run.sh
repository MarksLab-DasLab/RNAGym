#!/bin/bash

# the directory containing RNA-FM model
# downloaded from https://github.com/ml4bio/RNA-FM 
export model_location="path/to/model/location/RNA-FM/fm/"

export reference_sheet="path/to/reference_sheet.csv"
export dms_data_dir="path/to/dms/data/dir"
export output_scores_dir="path/to/output/scores/dir"

python compute_fitness.py  \
    --model_location "$model_location" \
    --reference_sequences "$reference_sheet" \
    --dms_directory "$dms_data_dir" \
    --output_directory "$output_scores_dir" 
