#!/usr/bin/env bash
set -euo pipefail

taplo format pixi.toml
ruff check --fix --extend-select I .
ruff format .
shfmt -w sh tests
