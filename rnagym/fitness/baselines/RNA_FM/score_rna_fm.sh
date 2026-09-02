#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
fitness_dir=$(cd -- "$script_dir/../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

row_id=${SLURM_ARRAY_TASK_ID:-0}

cd "$repository"
python -m rnagym.fitness.baselines.RNA_FM.score_rna_fm_single_dms \
	--row_id "$row_id" \
	--ref_sheet "$data_dir/reference_sheet_final.csv" \
	--dms_dir_path "$data_dir/assays" \
	--output_dir_path "$data_dir/model_predictions/rna_fm_4fill" \
	--checkpoint_path "$fitness_dir/.pixi/model-weights/rna-fm/RNA-FM_pretrained.pth"
