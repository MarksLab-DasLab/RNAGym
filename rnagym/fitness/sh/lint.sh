#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir/.."

taplo lint pixi.toml ../../pyproject.toml
ruff check --extend-select I . ../../tests/test_fitness.py ../config.py
ruff format --check . ../../tests/test_fitness.py ../config.py
shfmt -d sh baselines/*/*.sh
shellcheck sh/*.sh baselines/*/*.sh
