#!/usr/bin/env bash

set -euo pipefail

case ${1:-alphafold3} in
alphafold3)
	revision=231efc9bb9c13b45cc59e43f7107869084ee9624
	url=https://github.com/google-deepmind/alphafold3.git
	source_dir=.pixi/model-sources/alphafold3
	;;
rna-assessment)
	revision=d46f0472e7d52629283bb046a9c8f5d36b35f685
	url=https://github.com/RNA-Puzzles/RNA_assessment.git
	source_dir=.pixi/model-sources/RNA_assessment
	;;
*)
	echo "Unknown source: $1" >&2
	exit 2
	;;
esac

if [[ ! -d "$source_dir/.git" ]]; then
	git clone --no-checkout "$url" "$source_dir"
fi

git -C "$source_dir" fetch origin "$revision"
git -C "$source_dir" checkout --detach "$revision"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
[[ ${1:-} != rna-assessment ]] || chmod +x "$source_dir/MC-Annotate"
