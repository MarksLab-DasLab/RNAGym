#!/usr/bin/env bash

#SBATCH --job-name=rf2na
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres="gpu:1"

# Activate RF2NA environment
# NOTE(MCA): Add your RF2NA environment here
source "/path/to/conda/etc/profile.d/conda.sh"
conda activate RF2NA

# Parse arguments
JOB_DIR="$(pwd -P)"
LAUNCH_SH="$JOB_DIR/launch.sh"
echo "LAUNCH_DIR = '$LAUNCH_DIR'"
echo "LAUNCH_SCRIPT = '$LAUNCH_SH'"

# Run the script, capturing its exit code
echo "Running $LAUNCH_SH..."
bash "$LAUNCH_SH"
STATUS=$?

# If script succeeded, create SUCCESS file
if [ "$STATUS" -eq 0 ]; then
    touch SUCCESS
    echo "launch.sh completed successfully."
fi
exit "$STATUS"

