#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
fasta=$tmpdir/test.fa
printf '>test\nGGGGAAAACCCC\n' >"$fasta"

case $1 in
vienna | contrafold | eternafold | rnastructure)
	python -m tests.test_classical "$1"
	;;
ribonanzanet | ufold | rna-fm)
	CUDA_VISIBLE_DEVICES='' python -m "tests.test_${1//-/_}"
	;;
mxfold2)
	mxfold2 predict "$fasta" | grep -Eq '[().]{12}'
	;;
*)
	echo "unknown test: $1" >&2
	exit 2
	;;
esac
