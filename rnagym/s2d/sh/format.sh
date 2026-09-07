#!/usr/bin/env bash
set -euo pipefail

taplo format pixi.toml
ruff check --fix --extend-select I . ../config.py ../sequences.py ../../tests/test_s2d.py
ruff format . ../config.py ../sequences.py ../../tests/test_s2d.py
shfmt -w ../sh sh tasks/*.slurm
