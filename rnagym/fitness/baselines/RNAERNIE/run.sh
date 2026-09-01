#!/usr/bin/env bash
set -euo pipefail

: "${RNAERNIE_CHECKPOINT_DIR:?Set RNAERNIE_CHECKPOINT_DIR to the RNAErnie checkpoint directory}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

cd "$repository"
python -m rnagym.fitness.baselines.RNAERNIE.compute_fitness \
	--model_checkpoint "$RNAERNIE_CHECKPOINT_DIR" \
	--reference_sequences "$data_dir/reference_sheet_final.csv" \
	--dms_directory "$data_dir/assays" \
	--output_directory "$data_dir/model_predictions/RNAErnie" \
	--vocab_path "$script_dir/src/vocab_1MER.txt"
