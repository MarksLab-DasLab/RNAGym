#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
fitness_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

cd "$repository"
python -m rnagym.fitness.baselines.RNAERNIE.compute_fitness \
	--model_checkpoint "$fitness_dir/.pixi/model-weights/rna-ernie/checkpoint_final" \
	--reference_sequences "$data_dir/reference_sheet_final.csv" \
	--dms_directory "$data_dir/assays" \
	--output_directory "$data_dir/model_predictions/RNAErnie" \
	--vocab_path "$script_dir/src/vocab_1MER.txt"
