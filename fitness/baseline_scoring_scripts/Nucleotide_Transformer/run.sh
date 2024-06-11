#!/bin/bash
#SBATCH -n 1
#SBATCH -t 0-10:00
#SBATCH -p gpu,gpu_quad,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1,vram:40
#SBATCH --mem=20G
#SBATCH --output=/n/groups/marks/users/courtney/slurm/slurm_nt_score-%A_%3a-%x-%u.out
#SBATCH --error=/n/groups/marks/users/courtney/slurm/slurm_nt_score-%A_%3a-%x-%u.err
#SBATCH --job-name="slurm_nt_score"
#SBATCH --mail-user cshearer@g.harvard.edu
#SBATCH --array=1,2

#27,30

source ~/.bashrc

module load gcc/9.2.0
module load cuda/11.7

conda activate /n/groups/marks/software/anaconda_o2/envs/nucleotide_transformer

assay_id="${SLURM_ARRAY_TASK_ID}"

echo $assay_id

#python /n/groups/marks/users/courtney/projects/regulatory_genomics/rnagym/process_data.py ${assay_id}
python /n/groups/marks/users/courtney/projects/regulatory_genomics/rnagym/process_data.py ${assay_id}