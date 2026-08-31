#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

model_name=${NTV3_MODEL_NAME:-InstaDeepAI/NTv3_650M_pre}
prediction_name=${NTV3_PREDICTION_NAME:-ntv3_650m_4fill}
row_id=${SLURM_ARRAY_TASK_ID:-0}

cd "$repository"
python -m rnagym.fitness.baselines.Nucleotide_Transformer.compute_fitness \
	--row_id "$row_id" \
	--ref_sheet "$data_dir/reference_sheet_final.csv" \
	--dms_dir_path "$data_dir/assays" \
	--output_dir_path "$data_dir/model_predictions/$prediction_name" \
	--model_name "$model_name"
