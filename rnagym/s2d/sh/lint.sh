#!/usr/bin/env bash
set -euo pipefail

taplo lint pixi.toml
ruff check --extend-select I . ../config.py ../sequences.py ../../tests/test_s2d.py
ruff format --check . ../config.py ../sequences.py ../../tests/test_s2d.py
shfmt -d sh tasks/*.slurm
shellcheck sh/*.sh tasks/*.slurm
