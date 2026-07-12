#!/bin/bash

# Script used to score the Evo 2 models (evo2_40b by default; also evo2_7b, etc.)
# with the official evo2 package (https://github.com/ArcInstitute/evo2).
# Variants are scored by their mean per-token log-likelihood, averaged over the
# forward and reverse-complement strands (the evo2 baseline convention; pass
# --no-average_reverse_complement for forward strand only).
#
# evo2_40b requires FP8 via Transformer Engine on a Hopper GPU and does not fit
# on a single 80GB GPU; Vortex shards it across the visible GPUs, so select them
# with CUDA_VISIBLE_DEVICES and do not call .to(device).
export model_name="evo2_40b"

export reference_sheet="reference_sheet_final.csv"
# Write predictions under a folder named "evo2_40b" so they line up with the
# "evo2_40b" entry in merge_scoring_files.py (which reads the evo2_40b_score
# column from model_predictions/evo2_40b/).
export output_scores_dir="path/to/model_predictions/evo2_40b"
export dms_data_dir="path/to/dms/data/dir"
# Optional: path to a pre-merged evo2 checkpoint for fully-offline loading on
# air-gapped compute nodes (leave empty to download from HuggingFace).
export local_path=""

# Reference-sheet row to score. Set by a Slurm array job (0-69), or defaults
# to 0 when run directly.
DMS_index=${SLURM_ARRAY_TASK_ID:-0}

extra_args=()
[ -n "$local_path" ] && extra_args+=(--local_path "$local_path")

python score_evo2_single_dms.py \
    --row_id "$DMS_index" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --output_dir_path "$output_scores_dir" \
    --model_name "$model_name" \
    "${extra_args[@]}"
