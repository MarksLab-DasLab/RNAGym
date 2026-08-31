#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir/.."

taplo format pixi.toml ../../pyproject.toml
ruff check --fix --extend-select I . ../../tests/test_fitness.py ../config.py
ruff format . ../../tests/test_fitness.py ../config.py
shfmt -w sh baselines/*/*.sh
