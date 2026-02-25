#!/bin/bash

# Path to directory containing Rfam.cm and alignments/ folder
export rfam_dir="/n/lw_groups/hms/sysbio/marks/lab/databases/Rfam-15.0"
export ref_sheet="../../reference_sheet_final.csv"
export dms_dir="../../assays"
export out_dir="../../model_predictions/EVmutation"
export tmp_dir="./tmp_ev"

mkdir -p "$out_dir"
mkdir -p "$tmp_dir"

./compute_fitness.py \
    --rfam_dir "$rfam_dir" \
    --ref_sheet "$ref_sheet" \
    --dms_dir "$dms_dir" \
    --out_dir "$out_dir" \
    --tmp_dir "$tmp_dir" \
    --cpu 8
