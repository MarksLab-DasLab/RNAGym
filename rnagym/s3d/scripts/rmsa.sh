#!/bin/bash

#SBATCH --job-name=rmsa
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=65:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=general

set -euo pipefail

seq_unmod="$1"
seq_file="sequence.fa"

if [[ -f "SUCCESS" ]]; then
	exit 0
fi

echo "$seq_unmod" >"$seq_file"

/path/to/rMSA/rMSA.pl "$seq_file" -cpu=16 -fast=0

touch SUCCESS
