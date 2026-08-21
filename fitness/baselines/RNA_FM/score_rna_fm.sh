#!/bin/bash

# Script used to score RNA-FM (masked-marginal log-likelihood).
# RNA-FM is a ~100M parameter masked language model and fits on a single GPU.
# Requires the official package (pip install rna-fm) and the pretrained weights,
# RNA-FM_pretrained.pth, from the project's release.
export checkpoint_path="path/to/RNA-FM_pretrained.pth"

export reference_sheet="reference_sheet.csv"
# Write predictions under a folder named "rna_fm_4fill". One run writes all
# four fill strategies into it as RNA_FM_scores_wt_fill and so on, which the
# rna_fm_wt_fill, rna_fm_mask_fill, rna_fm_mut_fill and rna_fm_match_fill
# entries in fitness/merge_scoring_files.py read.
export output_scores_dir="path/to/model_predictions/rna_fm_4fill"
export dms_data_dir="path/to/dms/data/dir"

# Reference-sheet row to score, set by a Slurm array job or defaulting to 0.
# Submit the non-coding set with --array=0-8,11-32, which is the 31 ribozyme,
# tRNA and aptamer assays the v0.2 leaderboard reports; read the aggregate with
# performance_fitness.py --type ncRNA.
#
# The mRNA-coding and mRNA-splicing assays are NOT part of this leaderboard, and
# the four-strategy default does not run on them: their constructs exceed the
# model's position limit, and a windowed context drops mutations from the
# conditioning sequence, so the four fills would no longer estimate the same
# quantity. Score those one strategy at a time if you need them.

# Masked-marginal fill strategy. The default computes all four (wt-fill,
# mask-fill, mut-fill, match-fill), which share their contexts and so cost only
# about 19% more unique context examples than mut-fill alone, and writes one
# column per strategy named
# {COLUMN}_{strategy}. Pass --strategies mut-fill (or any single strategy) to
# write the historical bare {COLUMN} column instead. See
# fitness/baselines/masked_lm/strategies.py for the formulas.

DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python score_rna_fm_single_dms.py \
    --row_id "$DMS_index" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --output_dir_path "$output_scores_dir" \
    --checkpoint_path "$checkpoint_path"
