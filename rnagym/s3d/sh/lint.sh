#!/usr/bin/env bash
set -euo pipefail

taplo lint pixi.toml
ruff check --extend-select I .
ruff format --check .
shfmt -d sh scripts tasks
shellcheck -e SC1091 ./**/*.sh ./**/*.slurm
