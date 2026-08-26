#!/usr/bin/env bash

#SBATCH --job-name=trRNA
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=500G
#SBATCH --partition=gpu
#SBATCH --gres="gpu:1"

# Set conda environment
# NOTE(MCA): Add your path to trRNA conda environment here
source "/path/to/conda/etc/profile.d/conda.sh"
conda activate trRNA

# Launch trRNA prediction
set -euo pipefail
trRNA_dir="$1"
run_dir="$(pwd -P)"
echo "Launching trRNA prediction in $run_dir..."
#python3 "$trRNA_dir/predict.py" \
#    -i "$run_dir/sequence.a3m" \
#    -o "$run_dir/sequence.npz" \
#    -mdir "$trRNA_dir/params/model_1" \
#    -gpu 0

# Launch trRNA folding
python3 "$trRNA_dir/fold.py" \
    -npz "$run_dir/sequence.npz" \
    -fa "$run_dir/sequence.fa" \
    -out "$run_dir/model_1.pdb" \

touch SUCCESS
