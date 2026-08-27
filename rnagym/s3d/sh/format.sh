#!/usr/bin/env bash
set -euo pipefail

taplo format pixi.toml
ruff check --fix --extend-select I . ../config.py ../sequences.py ../tasks ../../tests/test_s3d.py
ruff format . ../config.py ../sequences.py ../tasks ../../tests/test_s3d.py
shfmt -w sh tasks/*.slurm
