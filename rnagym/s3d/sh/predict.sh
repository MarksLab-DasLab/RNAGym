#!/usr/bin/env bash

set -euo pipefail

# Slurm cluster configuration
cpu_partition=medium
gpu_partitions=gpu,gpu_advanced

environment=$1
python=".pixi/envs/$environment/bin/python"
mapfile -t kinds < <("$python" -m rnagym.s3d.tasks.predict "$environment" kinds)
# Setup may recover interrupted work, so never run it over active tasks
if squeue -h -u "$USER" -o %j | grep -q "^rg-$environment-"; then
	echo "$environment predictions are already queued"
	exit
fi
"$python" -m rnagym.s3d.tasks.predict "$environment" setup

mkdir -p .rg_jobs
for kind in "${kinds[@]}"; do
	if "$python" -m rnagym.s3d.tasks.predict "$environment" check "$kind"; then
		continue
	fi
	job_name="rg-$environment-${kind%s}"
	if squeue -h -u "$USER" -n "$job_name" | grep -q .; then
		echo "$environment $kind predictions are already queued"
		continue
	fi

	case "$environment:$kind" in
	af3:monomers)
		walltime=12:00:00 cpus=8 memory=64G gpu=h100
		;;
	af3:multimers)
		walltime=24:00:00 cpus=8 memory=200G gpu=h100
		;;
	nufold:monomers)
		walltime=12:00:00 cpus=4 memory=64G gpu=l40s
		;;
	rhofold:monomers)
		walltime=12:00:00 cpus=4 memory=128G gpu=h100
		;;
	rf2na:monomers)
		walltime=24:00:00 cpus=4 memory=64G gpu=l40s
		;;
	rf2na:multimers)
		walltime=24:00:00 cpus=4 memory=64G gpu=h100
		;;
	trrna:monomers)
		walltime=24:00:00 cpus=4 memory=128G gpu=l40s
		;;
	*)
		echo "No resources configured for $environment $kind" >&2
		exit 2
		;;
	esac

	dependency=()
	if [[ $environment == rf2na && $kind == multimers ]]; then
		prepare=$(sbatch --parsable --job-name=rg-rf2na-prepare --time=24:00:00 \
			--cpus-per-task=4 --mem=32G --partition="$cpu_partition" --array=0-63%8 \
			-- tasks/predict.slurm "$environment" prepare "$kind")
		dependency=(--dependency="afterok:$prepare")
	fi
	sbatch "${dependency[@]}" --job-name="$job_name" --time="$walltime" \
		--cpus-per-task="$cpus" --mem="$memory" --partition="$gpu_partitions" --gpus="$gpu:1" \
		--array=0-63%8 -- tasks/predict.slurm "$environment" predict "$kind"
done
