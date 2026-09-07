#!/usr/bin/env bash
set -euo pipefail
umask 002

checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$checkpoint_root/ntv3/hub"}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${NTV3_MODEL_NAME:-InstaDeepAI/NTv3_650M_pre}
prediction_name=${NTV3_PREDICTION_NAME:-ntv3_650m_4fill}
row_id=${1:-${SLURM_ARRAY_TASK_ID:-0}}

cd "$repository"
python -m rnagym.fitness.baselines.Nucleotide_Transformer.compute_fitness \
	--rows "$row_id" \
	--output "$data_dir/model_predictions/$prediction_name" \
	--model "$model_name"
