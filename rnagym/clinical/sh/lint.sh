#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir/.."

taplo lint pixi.toml ../../pyproject.toml
ruff check --extend-select I . ../../tests/test_clinical.py ../config.py
ruff format --check . ../../tests/test_clinical.py ../config.py
pyrefly check --python-interpreter-path "$PWD/.pixi/envs/default/bin/python" .
shfmt -d sh
shellcheck sh/*.sh
