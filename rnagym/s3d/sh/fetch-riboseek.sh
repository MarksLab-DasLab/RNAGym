#!/usr/bin/env bash

set -euo pipefail

version="1.0.1"
install_dir=".pixi/tools/riboseek-$version"
archive="riboseek-linux-gpu.tar.gz"
url="https://github.com/steineggerlab/riboseek/releases/download/v${version}/${archive}"
sha256="e77eb763ee6fdd577fa448066423a594201eb04c4b6a9ea9cd095669f51bdff0"

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
