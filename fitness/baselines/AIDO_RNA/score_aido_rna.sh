#!/bin/bash

# Script used to score the AIDO.RNA model (masked-marginal log-likelihood).
# AIDO.RNA-1.6B is an encoder-only masked language model and fits on a single
# 48 GB GPU in bfloat16 (peak around 10 GB at the default batch settings).
# Requires torch, transformers and the official GenBio model code:
#   pip install --no-deps modelgenerator
export model_name="genbio-ai/AIDO.RNA-1.6B"

export reference_sheet="reference_sheet.csv"
# Write predictions under a folder named "aido_rna" so they line up with the
# "aido_rna" entry in fitness/merge_scoring_files.py (which reads the
# aido_rna_score column from model_predictions/aido_rna/).
export output_scores_dir="path/to/model_predictions/aido_rna"
export dms_data_dir="path/to/dms/data/dir"

# Reference-sheet row to score. Set by a Slurm array job (0-69), or defaults
# to 0 when run directly. Rows 0-8,11-32 are the 31 ncRNA assays (ribozyme,
# tRNA, aptamer); scoring only those leaves aido_rna without mRNA predictions,
# so read its aggregate from performance_fitness.py --type ncRNA. Under
# --type all its All_Mean is NaN by design.
DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python score_aido_rna_single_dms.py \
    --row_id "$DMS_index" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --output_dir_path "$output_scores_dir" \
    --model_name "$model_name"
