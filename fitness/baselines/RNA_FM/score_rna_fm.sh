#!/bin/bash

# Script used to score RNA-FM (masked-marginal log-likelihood).
# RNA-FM is a ~100M parameter masked language model and fits on a single GPU.
# Requires the official package (pip install rna-fm) and the pretrained weights,
# RNA-FM_pretrained.pth, from the project's release.
export checkpoint_path="path/to/RNA-FM_pretrained.pth"

export reference_sheet="reference_sheet.csv"
# Write predictions under a folder named "RNA-FM" so they line up with the
# "RNA-FM" entry in fitness/merge_scoring_files.py (which reads the
# RNA_FM_scores column from model_predictions/RNA-FM/).
export output_scores_dir="path/to/model_predictions/RNA-FM"
export dms_data_dir="path/to/dms/data/dir"

# Reference-sheet row to score. Set by a Slurm array job (0-69), or defaults
# to 0 when run directly. Rows 0-8,11-32 are the 31 ncRNA assays; read an
# ncRNA-only aggregate with performance_fitness.py --type ncRNA.
DMS_index=${SLURM_ARRAY_TASK_ID:-0}

python score_rna_fm_single_dms.py \
    --row_id "$DMS_index" \
    --ref_sheet "$reference_sheet" \
    --dms_dir_path "$dms_data_dir" \
    --output_dir_path "$output_scores_dir" \
    --checkpoint_path "$checkpoint_path"
