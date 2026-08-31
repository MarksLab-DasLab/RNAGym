#!/usr/bin/env bash
set -euo pipefail

export COVERAGE_FILE="$PWD/.pixi/.coverage"
cd ..
python -m pytest -q tests/test_fitness.py \
	--cov=fitness.analyze_fill_strategies \
	--cov=fitness.baselines.Evo.score_evo2_single_dms \
	--cov=fitness/baselines/Nucleotide_Transformer \
	--cov=fitness/baselines/masked_lm \
	--cov=fitness.merge_scoring_files \
	--cov=fitness.model_registry \
	--cov=fitness.performance_fitness \
	--cov-fail-under=80 \
	--cov-report=term-missing
