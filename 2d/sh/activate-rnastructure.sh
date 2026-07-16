#!/usr/bin/env bash

# Arnie discovers this package as "RNAstructure" but looks it up internally as
# "rnastructure". An environment-local arniefile supplies the normalized key.
export ARNIEFILE="${CONDA_PREFIX}/arniefile.txt"
mkdir -p "${CONDA_PREFIX}/arnie-tmp"
printf 'rnastructure: %s/bin\nTMP: %s/arnie-tmp\n' \
	"${CONDA_PREFIX}" "${CONDA_PREFIX}" >"${ARNIEFILE}"
