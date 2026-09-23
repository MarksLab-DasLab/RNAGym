#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=../sh/weights.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/../../sh/weights.sh"

download_root=.pixi/downloads
source_root=.pixi/model-sources

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
	archive="$root/RF2NA_apr23.tgz"
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
		"$root/trRosettaRNA_v1.1.zip" \
		31b1a889819b1f04ebdf6857f521bf10750d49b59c3ad1b407fa43b204f608c4
	[[ -f "$source/predict.py" ]] || unzip -q \
		"$root/trRosettaRNA_v1.1.zip" -d "$source_root"
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
of3)
	root="$checkpoint_root/openfold3/of3-p2"
	download \
		https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-p2-155k.pt \
		"$root/of3-p2-155k.pt" \
		af09eac4f29cef856633af07558cb143226fe95ebbef2c20921769d4a5f4bee4
	link_latest openfold3 of3-p2
	;;
protenix)
	root="$checkpoint_root/protenix/v1.0.0/checkpoint"
	download \
		https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt \
		"$root/protenix_base_default_v1.0.0.pt" \
		2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04
	link_latest protenix v1.0.0
	;;
rf3)
	root="$checkpoint_root/rosettafold3/09-21"
	download \
		https://files.ipd.uw.edu/pub/rf3/rf3_foundry_09_21_preprint.ckpt \
		"$root/rf3_foundry_09_21_preprint.ckpt" \
		922901088366abb6e001bc5bd304f4002667fa7eb379cd901c94e4cac0bff762
	link_latest rosettafold3 09-21
	;;
boltz)
	root="$checkpoint_root/boltz/2.2.1"
	download \
		https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt \
		"$root/boltz2_conf.ckpt" \
		090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1
	# Boltz downloads the affinity head even when only folding structures
	download \
		https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt \
		"$root/boltz2_aff.ckpt" \
		dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e
	download \
		https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar \
		"$root/mols.tar" \
		39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7
	if [[ ! -d "$root/mols" ]]; then
		# Extract aside so an interrupted run cannot leave a partial cache
		rm -rf "$root/mols.tmp"
		mkdir -p "$root/mols.tmp"
		tar -xf "$root/mols.tar" -C "$root/mols.tmp"
		mv "$root/mols.tmp/mols" "$root/mols"
		rmdir "$root/mols.tmp"
	fi
	link_latest boltz 2.2.1
	;;
*)
	echo "Unknown model: $1" >&2
	exit 2
	;;
esac
