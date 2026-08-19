#!/bin/bash

# Script used to score RiNALMo (masked-marginal log-likelihood).
# RiNALMo giga-v1 is a 650M parameter masked language model and fits on a single
# GPU. Requires the official package (github.com/lbcb-sci/RiNALMo) and the
# giga-v1 checkpoint from the project's Zenodo record. The checkpoint stores the
# flash-attention module layout, so a CUDA device is required.
export checkpoint_path="path/to/rinalmo_giga_pretrained.pt"

export reference_sheet="reference_sheet.csv"
# Write predictions under a folder named "rinalmo_4fill". One run writes all
# four fill strategies into it as logit_scores_wt_fill and so on, which the
# rinalmo_wt_fill, rinalmo_mask_fill, rinalmo_mut_fill and rinalmo_match_fill
# entries in fitness/merge_scoring_files.py read.
export output_scores_dir="path/to/model_predictions/rinalmo_4fill"
export dms_data_dir="path/to/dms/data/dir"

# Reference-sheet row to score. Set by a Slurm array job (0-69), or defaults
# to 0 when run directly. Rows 0-8,11-32 are the 31 ncRNA assays; read an
# ncRNA-only aggregate with performance_fitness.py --type ncRNA.

# Masked-marginal fill strategy. The default computes all four (wt-fill,
# mask-fill, mut-fill, match-fill), which share their contexts and so cost only
# about 19% more unique context examples than mut-fill alone, and writes one
# column per strategy named
# {COLUMN}_{strategy}. Pass --strategies mut-fill (or any single strategy) to
# write the historical bare {COLUMN} column instead. See
# fitness/baselines/masked_lm/strategies.py for the formulas.

DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python score_rinalmo_single_dms.py \
    --row_id "$DMS_index" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --output_dir_path "$output_scores_dir" \
    --checkpoint_path "$checkpoint_path"
