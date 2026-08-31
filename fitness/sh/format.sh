#!/usr/bin/env bash
set -euo pipefail

taplo format pixi.toml
ruff check --fix --extend-select I \
	analyze_fill_strategies.py \
	baselines/Evo/score_evo2_single_dms.py \
	baselines/{AIDO_RNA,Orthrus,RNAGenesis,RNA_FM,RiNALMo}/*.py \
	baselines/masked_lm \
	merge_scoring_files.py \
	model_registry.py \
	performance_fitness.py \
	../tests/test_fitness.py
ruff format \
	analyze_fill_strategies.py \
	baselines/Evo/score_evo2_single_dms.py \
	baselines/{AIDO_RNA,Orthrus,RNAGenesis,RNA_FM,RiNALMo}/*.py \
	baselines/masked_lm \
	merge_scoring_files.py \
	model_registry.py \
	performance_fitness.py \
	../tests/test_fitness.py
shfmt -w sh baselines/{AIDO_RNA,Orthrus,RNAGenesis,RNA_FM,RiNALMo}/*.sh
