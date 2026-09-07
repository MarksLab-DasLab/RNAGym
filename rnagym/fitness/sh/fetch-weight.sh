#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=../sh/weights.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/../../sh/weights.sh"

case ${1:-} in
genslm)
	destination="$checkpoint_root/genslm/2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt"
	if ! printf '%s  %s\n' b13645dc523854a5f811d9a50bd747b611310d85d41a5498a9d7f93cea205f9d "$destination" | sha256sum --check --status 2>/dev/null; then
		echo "GenSLM publishes its 2.5B checkpoint through authenticated Globus:" >&2
		echo "https://app.globus.org/file-manager?origin_id=25918ad0-2a4e-4f37-bcfc-8183b19c3150" >&2
		echo "A verified copy is required at $destination" >&2
		exit 1
	fi
	;;
rna-ernie)
	destination="$checkpoint_root/rna-ernie/checkpoint_final"
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
		"$checkpoint_root/rna-fm/hub/checkpoints/RNA-FM_pretrained.pth" \
		5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99
	;;
rnagenesis)
	destination="$checkpoint_root/rnagenesis"
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
	temporary=$(mktemp -d "$checkpoint_root/.rnagenesis.XXXXXX")
	trap 'rm -rf -- "$temporary"' EXIT
	chmod 2775 "$temporary"
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
		"$checkpoint_root/rinalmo/rinalmo_giga_pretrained.pt" \
		cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db
	;;
*)
	echo "unknown weights: ${1:-}" >&2
	exit 2
	;;
esac
