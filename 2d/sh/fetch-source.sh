#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case $1 in
ribonanzanet)
	url=https://github.com/DasLab/rnet-inference.git
	revision=25996e720f25fc3c0c7e9679a45d54ff2d5f5500
	destination="$PROJECT_ROOT/.pixi/model-sources/rnet-inference"
	;;
ufold)
	url=https://github.com/uci-cbcl/UFold.git
	revision=75bd9acc83826059682dfca9d3659df66b132cd1
	destination="$PROJECT_ROOT/.pixi/model-sources/UFold"
	;;
eternafold)
	url=https://github.com/eternagame/EternaFold.git
	revision=702d3e485e768a6f2355d5d065e1241b04618e61
	destination="$PROJECT_ROOT/.pixi/model-sources/EternaFold"
	;;
*)
	echo "unknown source: $1" >&2
	exit 2
	;;
esac

mkdir -p "$(dirname "$destination")"

if [[ ! -d "$destination/.git" ]]; then
	git clone --no-checkout "$url" "$destination"
fi

git -C "$destination" remote set-url origin "$url"
git -C "$destination" fetch origin "$revision"
git -C "$destination" checkout --detach "$revision"
test "$(git -C "$destination" rev-parse HEAD)" = "$revision"

if [[ -f "$destination/.gitmodules" ]]; then
	git -C "$destination" submodule update --init --recursive
fi
