#!/usr/bin/env bash

set -euo pipefail

if python -c 'import alphafold3' 2>/dev/null; then
	exit
fi

python -m pip install --no-deps --no-build-isolation .pixi/model-sources/alphafold3
build_data
