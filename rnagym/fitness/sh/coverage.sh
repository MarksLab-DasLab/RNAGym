#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package_dir=$(cd -- "$script_dir/.." && pwd)
repository=$(cd -- "$package_dir/../.." && pwd)

export COVERAGE_FILE="$package_dir/.pixi/.coverage"
cd "$repository"
python -m pytest -q tests/test_fitness.py \
	--cov=rnagym.fitness.tasks.analyze_fill_strategies \
	--cov=rnagym.fitness.baselines.Evo.score_evo2_single_dms \
	--cov=rnagym.fitness.baselines.Nucleotide_Transformer \
	--cov=rnagym.fitness.baselines.masked_lm \
	--cov=rnagym.fitness.tasks.merge_scoring_files \
	--cov=rnagym.fitness.tasks.model_registry \
	--cov=rnagym.fitness.tasks.performance_fitness \
	--cov-fail-under=80 \
	--cov-report=term-missing
