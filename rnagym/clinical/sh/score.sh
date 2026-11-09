#!/usr/bin/env bash
# Score one shard of staged clinical files with one checkpoint, in that model's
# fitness environment, after the same setup as its fitness scoring wrapper.
# Usage: score.sh <model> <name> [<name> ...]. The shard is SLURM_ARRAY_TASK_ID
# (default 0) of NUM_SHARDS (default 1); SCORE_OPTIONS adds scorer options,
# such as --row-ids-dir or --out.
set -euo pipefail
umask 002

usage="usage: score.sh <model> <name> [<name> ...]"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$script_dir/../../.." && pwd)
fitness="$repository/rnagym/fitness"
checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
model=${1:?$usage}
shift
if (($# == 0)); then
	echo "$usage" >&2
	exit 2
fi

scorer=score_masked
case $model in
rna_fm) environment=rna-fm ;;
rinalmo) environment=rinalmo ;;
rnagenesis) environment=rnagenesis ;;
aido_rna | aido_rna_*) environment=aido-rna ;;
ntv3_*) environment=ntv3 ;;
orthrus) environment=orthrus ;;
evo2_*)
	environment=evo2
	scorer=score_evo2
	;;
*)
	echo "unknown model: $model" >&2
	exit 2
	;;
esac

run() {
	pixi run --manifest-path "$fitness/pixi.toml" --environment "$environment" "$@"
}

case $environment in
rna-fm | rinalmo | rnagenesis) run weights ;;
aido-rna | ntv3 | orthrus | evo2)
	export HF_HUB_CACHE=${HF_HUB_CACHE:-"$checkpoint_root/$environment/hub"}
	;;
esac
if [[ $environment == rinalmo ]]; then
	run source
	export PYTHONPATH="$fitness/.pixi/model-sources/RiNALMo${PYTHONPATH:+:$PYTHONPATH}"
fi

read -r -a options <<<"${SCORE_OPTIONS:-}"
cd "$repository"
run python -m "rnagym.clinical.$scorer" --model "$model" --names "$@" \
	--shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-1}" "${options[@]}"
