#!/usr/bin/env bash

#SBATCH --job-name=nufold
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres="gpu:1"

# Set conda environment
# NOTE(MCA): Add your path to nufold conda environment here
source "/path/to/conda/etc/profile.d/conda.sh"
conda activate nufold_P
export CUDA_HOME="$CONDA_PREFIX/pkgs/cuda-toolkit"

# Launch nufold
set -euo pipefail
nufold_dir="$1"
run_dir="$(pwd -P)"
echo "Launching nufold in $run_dir..."
mkdir -p output
python3 "$nufold_dir/run_nufold.py" \
    --ckpt_path "$nufold_dir/checkpoints/global_step145245.pt" \
    --input_fasta "$run_dir/input/"*"/"*".fasta" \
    --input_dir "$run_dir/input" \
    --output_dir "$run_dir/output" \
    --config_preset "initial_training"

touch SUCCESS
