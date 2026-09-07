#!/usr/bin/env bash
set -euo pipefail

taplo lint pixi.toml
ruff check --extend-select I . ../config.py ../sequences.py ../tasks ../../tests/test_s3d.py
ruff format --check . ../config.py ../sequences.py ../tasks ../../tests/test_s3d.py
shfmt -d ../sh sh tasks/*.slurm
shellcheck -e SC1091 ../sh/*.sh sh/*.sh tasks/*.slurm
