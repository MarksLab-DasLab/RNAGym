#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../../.." && pwd)
data_dir=${RNAGYM_DATA_DIR:-"$repository/data"}/fitness

cd "$repository"
python -m rnagym.fitness.baselines.EVmutation.compute_fitness \
	--msa_dir "$data_dir/msa/by_assay" \
	--ref_sheet "$data_dir/reference_sheet_final.csv" \
	--dms_dir "$data_dir/assays" \
	--out_dir "$data_dir/model_predictions/EVmutation" \
	--tmp_dir "$data_dir/tmp/evmutation" \
	--cpu "${SLURM_CPUS_PER_TASK:-8}"
