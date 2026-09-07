#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

rows=${1:-${SLURM_ARRAY_TASK_ID:-all}}

cd "$repository"
python -m rnagym.fitness.baselines.RNAGenesis.score_rnagenesis_single_dms \
	--rows "$rows" \
	--output "$data_dir/model_predictions/rnagenesis_4fill" \
	--model "$checkpoint_root/rnagenesis"
