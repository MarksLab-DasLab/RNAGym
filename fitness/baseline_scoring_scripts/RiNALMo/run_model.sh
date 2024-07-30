conda activate Rinalmo

# Get the current index from the array
DMS_index=${SLURM_ARRAY_TASK_ID}

python compute_fitness.py \
  --reference_sheet reference_sheet.csv \
  --task_id $DMS_index
