#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

case ${1:-} in
rinalmo)
	url=https://github.com/lbcb-sci/RiNALMo.git
	revision=2c2c5c14a5ae609d8c560a5d9ca32e51e0288955
	destination="$project_root/.pixi/model-sources/RiNALMo"
	;;
*)
	echo "unknown source: ${1:-}" >&2
	exit 2
	;;
esac

is_revision() {
	[[ -d $destination/.git ]] &&
		[[ $(git -C "$destination" rev-parse HEAD 2>/dev/null) == "$revision" ]]
}

if is_revision; then
	exit 0
fi
if [[ -e $destination ]]; then
	echo "$destination exists but is not revision $revision" >&2
	exit 1
fi

parent=$(dirname -- "$destination")
mkdir -p "$parent"
temporary=$(mktemp -d "$parent/.RiNALMo.XXXXXX")
trap 'rm -rf -- "$temporary"' EXIT

git clone --no-checkout "$url" "$temporary"
git -C "$temporary" checkout --detach "$revision"
test "$(git -C "$temporary" rev-parse HEAD)" = "$revision"

if ! mv -T -- "$temporary" "$destination" 2>/dev/null && ! is_revision; then
	echo "Could not install RiNALMo source at $destination" >&2
	exit 1
fi
trap - EXIT
