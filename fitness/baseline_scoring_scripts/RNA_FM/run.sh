#!/bin/bash
#SBATCH -c 3                               # Request one core
#SBATCH -t 0-05:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=5G                           # Memory total in MiB (for all cores)
#SBATCH -o hostname_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e hostname_%j.err                 # File to which STDERR will be written, including job ID (%j)


python compute_fitness.py  \
    --model_location /n/groups/marks/projects/rna_structure/code/RNA-FM/fm/ \
    --reference_sequences /n/groups/marks/projects/RNAgym/mutational_assays/reference_sheet_cleaned_CAS.csv \
    --dms_directory /n/groups/marks/projects/RNAgym/mutational_assays/processed_gdrive/ \
    --output_directory /n/groups/marks/projects/RNAgym/baselines/RNA-FM/test-batch/test_run/ 
