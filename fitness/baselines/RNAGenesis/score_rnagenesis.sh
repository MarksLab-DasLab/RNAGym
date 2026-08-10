#!/bin/bash

# Script used to score the RNAGenesis encoder (masked-marginal log-likelihood).
# The released encoder is a ~709M parameter xTrimoPGLM-style masked LM and fits
# on a single GPU in bfloat16 (peak under 3 GB at the default batch settings).
#
# Setup notes for the checkpoint at https://huggingface.co/Zaixi/RNAGenesis:
#  - access is gated, so accept the licence and run `hf auth login` first
#  - the repo omits quantization.py even though modeling_xtrimopglm.py imports
#    it, so copy that file in from the same upstream codebase
#    (biomap-research/xtrimopglm-1b-mlm) and point --model_name at the local copy
#  - the modelling code imports deepspeed at module level, so deepspeed must be
#    installed even though only inference is used
export model_dir="path/to/local/RNAGenesis"

export reference_sheet="reference_sheet.csv"
# Write predictions under a folder named "rnagenesis" so they line up with the
# "rnagenesis" entry in fitness/merge_scoring_files.py (which reads the
# rnagenesis_score column from model_predictions/rnagenesis/).
export output_scores_dir="path/to/model_predictions/rnagenesis"
export dms_data_dir="path/to/dms/data/dir"

# Reference-sheet row to score. Set by a Slurm array job (0-69), or defaults
# to 0 when run directly. Rows 0-8,11-32 are the 31 ncRNA assays (ribozyme,
# tRNA, aptamer); scoring only those leaves rnagenesis without mRNA predictions,
# so read its aggregate from performance_fitness.py --type ncRNA. Under
# --type all its All_Mean is NaN by design.
DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python score_rnagenesis_single_dms.py \
    --row_id "$DMS_index" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --output_dir_path "$output_scores_dir" \
    --model_name "$model_dir"
