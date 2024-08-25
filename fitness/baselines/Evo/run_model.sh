#!/bin/bash

export RNA_DATA_DIR='Path_to_DMS_assays_fasta'
export RNA_RESULTS_DIR='Path_to_model_prediction'
export fitness_filename='DMS_filename'
export reference_sheet="reference_sheet.csv"
export DMS_index=0

python -m scripts.score \
    --input-fasta ${RNA_DATA_DIR}/${fitness_filename}.fasta \
    --output-tsv ${RNA_RESULTS_DIR}/${fitness_filename}.tsv \
    --model-name 'evo-1-131k-base' \
    --device cuda:0

python compute_fitness.py ${reference_sheet} ${DMS_index} ${RNA_RESULTS_DIR}/${fitness_filename}.tsv
