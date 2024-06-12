#!/bin/bash
#SBATCH -n #
#SBATCH --gres=#gpu that supports flash attention
#SBATCH -p #
#SBATCH -t #
#SBATCH --mem=25G
#SBATCH --output=#
#SBATCH --error=#
#SBATCH --job-name=#
#SBATCH --array=0-31

# Get the current index from the array
DMS_index=${SLURM_ARRAY_TASK_ID}

python compute_fitness.py \
  --reference_sheet reference_sheet.csv \
  --task_id $DMS_index