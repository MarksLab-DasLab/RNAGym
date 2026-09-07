#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package_dir=$(cd -- "$script_dir/.." && pwd)
repository=$(cd -- "$package_dir/../.." && pwd)

coverage_bootstrap=$(mktemp -d)
trap 'rm -rf -- "$coverage_bootstrap"' EXIT
# Start coverage in each model environment without modifying its installation
printf '%s\n' 'import coverage' 'coverage.process_startup()' >"$coverage_bootstrap/sitecustomize.py"
export PYTHONPATH="$coverage_bootstrap:$repository${PYTHONPATH:+:$PYTHONPATH}"
export COVERAGE_FILE="$package_dir/.pixi/.coverage"
export COVERAGE_RCFILE="$repository/pyproject.toml"
cd "$repository"
python -m coverage erase
python -m coverage run -m pytest -q tests/test_fitness.py
status=0
python -m coverage run -m rnagym.fitness.tasks.check_published "$@" || status=$?
python -m coverage combine
python -m coverage report --show-missing --fail-under=80 || status=$?
python -m coverage xml -o "$package_dir/.pixi/coverage.xml"
exit "$status"
