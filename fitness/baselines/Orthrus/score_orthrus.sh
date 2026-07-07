#!/bin/bash

# Script used to score the Orthrus MLM model (masked-marginal log-likelihood).
# Orthrus is a small (~10M param) Mamba RNA model and fits on a single GPU.
# Requires the Mamba kernels (causal-conv1d, mamba-ssm) + transformers with
# trust_remote_code (the model class ships in the HF repo).
export model_name="antichronology/orthrus-mlm-6-track"

export reference_sheet="reference_sheet.csv"
# Write predictions under a folder named "orthrus" so they line up with the
# "orthrus" entry in fitness/merge_scoring_files.py (which reads the
# orthrus_score column from model_predictions/orthrus/).
export output_scores_dir="path/to/model_predictions/orthrus"
export dms_data_dir="path/to/dms/data/dir"

# Reference-sheet row to score. Set by a Slurm array job (0-69), or defaults
# to 0 when run directly.
DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python score_orthrus_single_dms.py \
    --row_id "$DMS_index" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --output_dir_path "$output_scores_dir" \
    --model_name "$model_name"
