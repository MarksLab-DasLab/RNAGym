#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=../sh/slurm.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/../../sh/slurm.sh"

# Slurm cluster configuration
cpu_partition=medium
gpu_partitions=gpu,gpu_advanced

environment=$1
# Jobs are named per alignment source so runs against different alignments
# neither collide in the queue nor mistake each other for already queued work
source_tag=""
if [[ ${RNAGYM_MSA_SOURCE:-riboseek} != riboseek ]]; then
	source_tag="-${RNAGYM_MSA_SOURCE}"
fi
# Setup builds native extensions, which need the environment's own toolchain
export PATH="$PWD/.pixi/envs/$environment/bin:$PATH"
# rnagym/s3d/cmd shadows the standard library, but the legacy predictors run
# their own source trees as scripts and need that directory on the path
case $environment in
af3 | boltz | of3 | protenix | rf3)
	export PYTHONSAFEPATH=1
	;;
esac
python=".pixi/envs/$environment/bin/python"
mapfile -t kinds < <("$python" -m rnagym.s3d.tasks.predict "$environment" kinds)
# Setup may recover interrupted work, so never run it over active tasks
# Match whole names, since rg-af3- also prefixes rg-af3-mmseqs- jobs
if squeue -h -u "$USER" -o %j | grep -qxE "rg-$environment$source_tag-(monomer|multimer|prepare)"; then
	echo "$environment predictions are already queued"
	exit
fi
"$python" -m rnagym.s3d.tasks.predict "$environment" setup

mkdir -p .rg_jobs
for kind in "${kinds[@]}"; do
	job_name="rg-$environment$source_tag-${kind%s}"
	# Freeze the missing targets, so no task starts idle and none drops or
	# repeats a target as others finish
	targets=".rg_jobs/$job_name.targets"
	rm -f "$targets"
	if "$python" -m rnagym.s3d.tasks.predict "$environment" check "$kind" "$targets"; then
		continue
	fi
	tasks=$(wc -l <"$targets")
	array="0-$((tasks < 64 ? tasks - 1 : 63))%8"

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
	of3:monomers | protenix:monomers | rf3:monomers | boltz:monomers)
		walltime=12:00:00 cpus=8 memory=64G gpu=h100
		;;
	of3:multimers | protenix:multimers | rf3:multimers | boltz:multimers)
		walltime=24:00:00 cpus=8 memory=200G gpu=h100
		;;
	*)
		echo "No resources configured for $environment $kind" >&2
		exit 2
		;;
	esac

	dependency=$(gpu_array_dependency)
	if [[ $environment == rf2na && $kind == multimers ]]; then
		prepare=$(sbatch --parsable --job-name="rg-rf2na$source_tag-prepare" --time=24:00:00 \
			--cpus-per-task=4 --mem=32G --partition="$cpu_partition" --array="$array" \
			-- tasks/predict.slurm "$environment" prepare "$kind" "$targets")
		dependency+="${dependency:+,}afterok:$prepare"
	fi
	sbatch ${dependency:+"--dependency=$dependency"} --job-name="$job_name" --time="$walltime" \
		--cpus-per-task="$cpus" --mem="$memory" --partition="$gpu_partitions" --gpus="$gpu:1" \
		--array="$array" -- tasks/predict.slurm "$environment" predict "$kind" "$targets"
done
