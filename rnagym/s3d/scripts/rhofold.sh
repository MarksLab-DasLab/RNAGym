#!/usr/bin/env bash

#SBATCH --job-name=rhofold
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres="gpu:1"

# Set conda environment
# NOTE(MCA): Add your path to Rhofold+ conda environment here
source "/path/to/conda/etc/profile.d/conda.sh"
conda activate rhofold

# Launch Rhofold
set -euo pipefail
rhofold_dir="$1"
run_dir="$(pwd -P)"
echo "Launching RhoFold in $run_dir..."
mkdir -p output
python3 "$rhofold_dir/inference.py" \
	--input_fas "$run_dir/sequence.fa" \
	--input_a3m "$run_dir/sequence.a3m" \
	--output_dir "$run_dir/output" \
	--ckpt "$rhofold_dir/pretrained/RhoFold_pretrained.pt" \
	--device "cuda:0"

touch SUCCESS
