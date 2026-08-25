#!/usr/bin/env bash

#SBATCH --account=marks_dsm1
#SBATCH --output=.rg_jobs/%x-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
environment=$1
shift
export PATH="$PWD/.pixi/envs/$environment/bin:$PATH"

if [[ $environment == af3 ]]; then
	# Use host memory when an input exceeds GPU memory
	export XLA_PYTHON_CLIENT_PREALLOCATE=false
	export TF_FORCE_UNIFIED_MEMORY=true
	export XLA_CLIENT_MEM_FRACTION=3.2
	export PYTHONSAFEPATH=1
elif [[ $environment == rf2na ]]; then
	# RF2NA imports sibling source trees as top-level modules
	source_dir="$PWD/.pixi/model-sources/RoseTTAFold2NA"
	export PYTHONPATH="$source_dir/network:$source_dir/SE3Transformer"
	export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
fi
".pixi/envs/$environment/bin/python" -m rnagym.s3d.tasks.predict \
	"$environment" "$@" \
	"$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT"
