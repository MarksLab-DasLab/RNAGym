#!/usr/bin/env bash
set -euo pipefail

taplo lint pixi.toml
ruff check --extend-select I \
	analyze_fill_strategies.py \
	baselines/Evo/score_evo2_single_dms.py \
	baselines/{AIDO_RNA,Nucleotide_Transformer,Orthrus,RNAGenesis,RNA_FM,RiNALMo}/*.py \
	baselines/masked_lm \
	merge_scoring_files.py \
	model_registry.py \
	performance_fitness.py \
	../tests/test_fitness.py
ruff format --check \
	analyze_fill_strategies.py \
	baselines/Evo/score_evo2_single_dms.py \
	baselines/{AIDO_RNA,Nucleotide_Transformer,Orthrus,RNAGenesis,RNA_FM,RiNALMo}/*.py \
	baselines/masked_lm \
	merge_scoring_files.py \
	model_registry.py \
	performance_fitness.py \
	../tests/test_fitness.py
shfmt -d sh baselines/{AIDO_RNA,Nucleotide_Transformer,Orthrus,RNAGenesis,RNA_FM,RiNALMo}/*.sh
shellcheck sh/*.sh baselines/{AIDO_RNA,Nucleotide_Transformer,Orthrus,RNAGenesis,RNA_FM,RiNALMo}/*.sh
