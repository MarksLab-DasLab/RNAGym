#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
weight_root="$project_root/.pixi/model-weights"
mkdir -p "$weight_root"
exec 9>"$weight_root/.download.lock"
flock 9

download() {
	local url=$1 destination=$2 checksum=$3 temporary

	mkdir -p "$(dirname -- "$destination")"
	if ! printf '%s  %s\n' "$checksum" "$destination" | sha256sum --check --status 2>/dev/null; then
		temporary=$(mktemp "${destination}.XXXXXX")
		if ! curl -L --fail --retry 3 --output "$temporary" "$url"; then
			rm -f -- "$temporary"
			return 1
		fi
		if ! printf '%s  %s\n' "$checksum" "$temporary" | sha256sum --check --status; then
			rm -f -- "$temporary"
			return 1
		fi
		mv -- "$temporary" "$destination"
	fi
}

case ${1:-} in
genslm)
	destination="$project_root/.pixi/model-weights/genslm/2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt"
	if [[ ! -f $destination ]]; then
		echo "GenSLM publishes its 2.5B checkpoint through authenticated Globus:" >&2
		echo "https://app.globus.org/file-manager?origin_id=25918ad0-2a4e-4f37-bcfc-8183b19c3150" >&2
		echo "Download it to $destination" >&2
		exit 1
	fi
	;;
rna-ernie)
	destination="$project_root/.pixi/model-weights/rna-ernie/checkpoint_final"
	download \
		'https://drive.usercontent.google.com/download?id=1MQjtnrtssoF5qAiALakaDDBnQfy0PJC4&export=download&confirm=t' \
		"$destination/model_state.pdparams" \
		e5f17a98af4f1051cabc929b4b1e7d20a8bc8f989b296ef1b515e19b19a3867b
	download \
		https://raw.githubusercontent.com/soberTTT/RNAErnie-1/b5e4c1cfdc6f53101111823abaddcab00c43b0d3/output/BERT%2CERNIE%2CMOTIF%2CPROMPT/checkpoint_final/model_config.json \
		"$destination/model_config.json" \
		4fe00781c61d30151a797e902dbbc68d9c3937436b9f67fa478c563f4b6d8d2b
	;;
rna-fm)
	download \
		https://huggingface.co/cuhkaih/rnafm/resolve/main/RNA-FM_pretrained.pth \
		"$project_root/.pixi/model-weights/rna-fm/RNA-FM_pretrained.pth" \
		5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
	;;
rnagenesis)
	destination="$weight_root/rnagenesis"
	if [[ -s $destination/config.json &&
		-s $destination/configuration_xtrimopglm.py &&
		-s $destination/modeling_xtrimopglm.py &&
		-s $destination/pytorch_model.bin &&
		-s $destination/tokenizer.model ]] &&
		printf '%s  %s\n' \
			7a690346ab3866ebdc361fb5d7389ecf6e3e8fe62f50868d28a5bcbff52ecd20 \
			"$destination/quantization.py" | sha256sum --check --status 2>/dev/null; then
		exit 0
	fi
	if [[ -e $destination ]]; then
		echo "$destination exists but is not a complete RNAGenesis checkpoint" >&2
		exit 1
	fi
	temporary=$(mktemp -d "$weight_root/.rnagenesis.XXXXXX")
	trap 'rm -rf -- "$temporary"' EXIT
	hf download Zaixi/RNAGenesis \
		--revision d8a42130984cbf04f6a5e16a3aa0c0d6578036a8 \
		--local-dir "$temporary"
	# RNAGenesis imports this xTrimoPGLM file but omits it from its model repo
	download \
		https://huggingface.co/biomap-research/xtrimopglm-1b-mlm/resolve/676c1bcb9f737c166a25587f83fd580610acb1ad/quantization.py \
		"$temporary/quantization.py" \
		7a690346ab3866ebdc361fb5d7389ecf6e3e8fe62f50868d28a5bcbff52ecd20
	mv -T -- "$temporary" "$destination"
	trap - EXIT
	;;
rinalmo)
	download \
		https://zenodo.org/api/records/15043668/files/rinalmo_giga_pretrained.pt/content \
		"$project_root/.pixi/model-weights/rinalmo/rinalmo_giga_pretrained.pt" \
		cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db
	;;
*)
	echo "unknown weights: ${1:-}" >&2
	exit 2
	;;
esac
