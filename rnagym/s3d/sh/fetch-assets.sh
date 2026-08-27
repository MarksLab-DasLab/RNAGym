#!/usr/bin/env bash

set -euo pipefail

checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
download_root=.pixi/downloads
source_root=.pixi/model-sources

download() {
	local url=$1 destination=$2 checksum=$3

	mkdir -p "$(dirname "$destination")"
	if ! echo "$checksum  $destination" | sha256sum --check --status 2>/dev/null; then
		curl -L --fail --retry 3 --output "$destination.tmp" "$url"
		echo "$checksum  $destination.tmp" | sha256sum --check --status
		mv "$destination.tmp" "$destination"
	fi
}

link_latest() {
	local model=$1 version=$2

	mkdir -p "$checkpoint_root/$model"
	ln -sfn "$version" "$checkpoint_root/$model/latest"
}

install_ipknot() {
	local archive="$download_root/ipknot-1.1.0-x86_64-linux.zip"

	download \
		https://github.com/satoken/ipknot/releases/download/v1.1.0/ipknot-1.1.0-x86_64-linux.zip \
		"$archive" \
		daa2e5cb88684222c3bc2f3b356330f903652ffb9896d7c3a254c8b29d6424cc
	if [[ ! -x "$source_root/ipknot/ipknot-1.1.0-x86_64-linux/ipknot" ]]; then
		mkdir -p "$source_root/ipknot"
		unzip -q "$archive" -d "$source_root/ipknot"
	fi
}

case $1 in
nufold)
	root="$checkpoint_root/nufold/v1.0"
	download https://kiharalab.org/nufold/global_step145245.pt \
		"$root/global_step145245.pt" \
		7e473b98e727c9ae3a88fecd917262fe115f53a9f9c098db965baebe597cfbdb
	install_ipknot
	link_latest nufold v1.0
	;;
rhofold)
	root="$checkpoint_root/rhofold/2023-12-31"
	download \
		https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt \
		"$root/rhofold_pretrained_params.pt" \
		3adb621978dfcd7ea0dc0edeb520d249f423d61530df85ff58a1fad2f33e5608
	link_latest rhofold 2023-12-31
	;;
rf2na)
	root="$checkpoint_root/rosettafold2na/0.2"
	archive="$download_root/RF2NA_apr23.tgz"
	download https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz \
		"$archive" \
		1a5e9f6dc6fde298f883e4d58d7766d464053f200c5247ccf18f5cc8dfdb3809
	mkdir -p "$root"
	if ! echo "fadbb086542b00dd687cb03098a200e105bf37463c82e259f00c52bd03e5815c  $root/RF2NA_apr23.pt" | sha256sum --check --status 2>/dev/null; then
		tar -xOf "$archive" weights/RF2NA_apr23.pt >"$root/RF2NA_apr23.pt.tmp"
		echo "fadbb086542b00dd687cb03098a200e105bf37463c82e259f00c52bd03e5815c  $root/RF2NA_apr23.pt.tmp" | sha256sum --check --status
		mv "$root/RF2NA_apr23.pt.tmp" "$root/RF2NA_apr23.pt"
	fi
	link_latest rosettafold2na 0.2
	;;
trrna)
	root="$checkpoint_root/trrosettarna/v1.1"
	source="$source_root/trRosettaRNA_v1.1"
	download \
		https://yanglab.qd.sdu.edu.cn/trRosettaRNA/download/trRosettaRNA_v1.1.zip \
		"$download_root/trRosettaRNA_v1.1.zip" \
		31b1a889819b1f04ebdf6857f521bf10750d49b59c3ad1b407fa43b204f608c4
	[[ -f "$source/predict.py" ]] || unzip -q \
		"$download_root/trRosettaRNA_v1.1.zip" -d "$source_root"
	if [[ ! -d "$root/params" ]]; then
		mkdir -p "$root"
		mv "$source/params" "$root/params"
	fi
	if [[ -e "$source/params" && ! -L "$source/params" ]]; then
		rm -r "$source/params"
	fi
	ln -sfn "$root/params" "$source/params"
	install_ipknot
	link_latest trrosettarna v1.1
	;;
*)
	echo "Unknown model: $1" >&2
	exit 2
	;;
esac
