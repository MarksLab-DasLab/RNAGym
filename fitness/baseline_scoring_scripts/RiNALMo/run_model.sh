#!/bin/bash
#SBATCH -n 2
#SBATCH --gres=gpu:a40:1
#SBATCH -p gpu_quad 
#SBATCH -t 0-08:00
#SBATCH --mem=25G
#SBATCH --output=slurm/slurm_Rinalmo%j.out
#SBATCH --error=slurm/slurm_Rinalmo%j.err
#SBATCH --job-name=Rinalmo
#SBATCH --array=0-31

module load gcc/9.2.0
module load cuda/11.7

source /n/app/miniconda3/23.1.0/bin/activate
conda activate Rinalmo

# Get the current index from the array
DMS_index=${SLURM_ARRAY_TASK_ID}

python compute_fitness_mm_fix_2.py \
  --reference_sheet reference_sheet.csv \
  --task_id $DMS_index