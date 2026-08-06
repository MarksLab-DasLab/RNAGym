#!/usr/bin/env bash
set -euo pipefail

download() {
	local url=$1 destination=$2 checksum=$3

	mkdir -p "$(dirname "$destination")"
	if ! echo "$checksum  $destination" | sha256sum --check --status 2>/dev/null; then
		curl -L --fail --retry 3 --output "$destination.tmp" "$url"
		echo "$checksum  $destination.tmp" | sha256sum --check --status
		mv "$destination.tmp" "$destination"
	fi
}

case $1 in
ufold)
	download \
		'https://drive.usercontent.google.com/download?id=1DJDkKwZNdt-cwKPkDxu3ErWqoTK2k7uy&export=download&confirm=t' \
		.pixi/model-weights/ufold/ufold_train_alldata.pt \
		ea84411dd59f230b0a79d984c116dad8c8a75bef1b9d0079376b581d0cdae64b
	;;
rna-fm)
	download \
		https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth \
		.pixi/model-weights/rna-fm/hub/checkpoints/RNA-FM_pretrained.pth \
		5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
	download \
		https://huggingface.co/cuhkaih/rnafm/resolve/main/SS/RNA-FM-ResNet_PDB-All.pth \
		.pixi/model-weights/rna-fm/RNA-FM-ResNet_PDB-All.pth \
		bc9df2111ae6ba95b373641ed1ab35ef4fa8b4cd0f8d21065a55450f575fe15d
	;;
*)
	echo "unknown weights: $1" >&2
	exit 2
	;;
esac
