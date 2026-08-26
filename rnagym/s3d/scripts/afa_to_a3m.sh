#!/usr/bin/env bash

###############################################################################
# `afa_to_a3m.sh`: Converts AFA sequence alignments to A3M.
###############################################################################
mapfile -t afa_files < <(find ./out -maxdepth 4 -type f -name "sequence.afa")

for alignment in "${afa_files[@]}"; do
    output="${alignment%.afa}.a3m"
    echo "Converting $alignment -> $output..."

    (reformat.pl fas a3m "$alignment" "$output" \
	|| echo "Error converting $alignment" >&2) &
done

