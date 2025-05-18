#!/usr/bin/env bash

#SBATCH --job-name=af3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres="gpu:1"

# Set conda environment
source "/path/to/conda/etc/profile.d/conda.sh"
conda activate af3

# Parse command line
af3_dir="$1"
db_dir="$af3_dir/db"
model_dir="$af3_dir/params"
output_dir="$(pwd -P)"
json_path="$output_dir/config.json"

# Print debug info
echo "af3_dir is: '$af3_dir'"
echo "db_dir is: '$db_dir'"
echo "model_dir is: '$model_dir'"
echo "output_dir is: '$output_dir'"
echo "json_path is: '$json_path'"

# Launch AF3, failing on error
set -euo pipefail
echo "Launching AF3 on $json_path..."
python3 "$af3_dir/alphafold3/run_alphafold.py" \
    --db_dir="$db_dir" \
    --model_dir="$model_dir" \
    --json_path="$json_path" \
    --output_dir="$output_dir"

# Note successful completion
touch SUCCESS
