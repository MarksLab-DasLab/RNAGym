#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=../sh/weights.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/../../sh/weights.sh"

case $1 in
eternafold)
	download \
		https://raw.githubusercontent.com/eternagame/EternaFold/702d3e485e768a6f2355d5d065e1241b04618e61/parameters/EternaFoldParams.v1 \
		"$checkpoint_root/eternafold/EternaFoldParams.v1" \
		1421e89cc8df24b53a320eff6c72b8acfdb2771faba29ab9c798f7bdfc853a8a
	;;
mxfold2)
	download \
		https://raw.githubusercontent.com/keio-bioinformatics/mxfold2/51b213676708bebd664f0c40873a46e09353e1ee/mxfold2/models/TrainSetAB.pth \
		"$checkpoint_root/mxfold2/TrainSetAB.pth" \
		8ba1283e70c1c073033161adc120f67cdc467176476358d26a12cadaf63262ec
	;;
ribonanzanet)
	download \
		https://raw.githubusercontent.com/DasLab/rnet-inference/25996e720f25fc3c0c7e9679a45d54ff2d5f5500/RibonanzaNet-Weights/RibonanzaNet-SS.pt \
		"$checkpoint_root/ribonanzanet/RibonanzaNet-SS.pt" \
		626060952368affbf61b78b532d6166387094754b68bc0553da376f2d2b00d56
	;;
rinalmo)
	download \
		https://zenodo.org/api/records/15043668/files/rinalmo_giga_ss_bprna_ft.pt/content \
		"$checkpoint_root/rinalmo/rinalmo_giga_ss_bprna_ft.pt" \
		44e377cb0c92f7b9cff8db8a38e33e5c0363d2f9a19831c6342346bb788a5e55
	;;
ufold)
	download \
		'https://drive.usercontent.google.com/download?id=1DJDkKwZNdt-cwKPkDxu3ErWqoTK2k7uy&export=download&confirm=t' \
		"$checkpoint_root/ufold/ufold_train_alldata.pt" \
		ea84411dd59f230b0a79d984c116dad8c8a75bef1b9d0079376b581d0cdae64b
	;;
rna-fm)
	download \
		https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth \
		"$checkpoint_root/rna-fm/hub/checkpoints/RNA-FM_pretrained.pth" \
		5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
	download \
		https://huggingface.co/cuhkaih/rnafm/resolve/main/SS/RNA-FM-ResNet_PDB-All.pth \
		"$checkpoint_root/rna-fm/RNA-FM-ResNet_PDB-All.pth" \
		bc9df2111ae6ba95b373641ed1ab35ef4fa8b4cd0f8d21065a55450f575fe15d
	;;
*)
	echo "unknown weights: $1" >&2
	exit 2
	;;
esac
