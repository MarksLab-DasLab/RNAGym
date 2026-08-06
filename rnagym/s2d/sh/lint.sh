#!/usr/bin/env bash
set -euo pipefail

taplo lint pixi.toml
ruff check --extend-select I .
ruff format --check .
shfmt -d sh tests
shellcheck ./**/*.sh ./**/*.slurm
