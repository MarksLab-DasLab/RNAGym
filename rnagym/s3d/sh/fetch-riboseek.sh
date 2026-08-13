#!/usr/bin/env bash

set -euo pipefail

version="1.0.0"
install_dir=".pixi/tools/riboseek"
archive="riboseek-linux-gpu.tar.gz"
url="https://github.com/steineggerlab/riboseek/releases/download/v${version}/${archive}"
sha256="0492b80d27dbc5da1d085e458618236fe58060a68e10d22ace26d045034af6bd"

if [[ ! -x "$install_dir/bin/riboseek" ]]; then
	temporary=$(mktemp -d)
	trap 'rm -rf "$temporary"' EXIT
	curl -L --fail "$url" -o "$temporary/$archive"
	echo "$sha256  $temporary/$archive" | sha256sum --check
	tar -xzf "$temporary/$archive" -C "$temporary"
	mkdir -p "$(dirname "$install_dir")"
	mv "$temporary/riboseek" "$install_dir"
fi

ln -sfn "$(realpath "$install_dir/bin/riboseek")" "$CONDA_PREFIX/bin/riboseek"
