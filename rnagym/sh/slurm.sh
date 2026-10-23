#!/usr/bin/env bash

# RNAGym GPU arrays run at most 8 tasks at a time, and each new one waits for
# the third newest still queued, so together they hold at most 24 GPUs
gpu_array_dependency() {
	local -a arrays
	mapfile -t arrays < <(
		squeue -h -u "$USER" -o "%F %j %P" | awk '$2 ~ /^rg-/ && $3 ~ /gpu/ {print $1}' | sort -nu
	)
	if ((${#arrays[@]} >= 3)); then
		echo "afterany:${arrays[-3]}"
	fi
}
