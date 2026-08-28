#!/bin/sh

set -eu

script_dir="$(CDPATH= cd "$(dirname "$0")" && pwd)"

cleanup() {
    rm -f "$notebook" "${notebook}.wal"
    rmdir "$temp_dir"
}

# DuckDB stores notebooks in this database rather than as standalone files
ui_dir="${HOME}/.duckdb/extension_data/ui"
# DuckDB rejects existing empty files, so use a new path in a temporary directory
temp_dir="$(mktemp -d)"
notebook="${temp_dir}/ui.db"
trap cleanup EXIT HUP INT TERM

duckdb "$notebook" < "${script_dir}/browse.sql"
mkdir -p "$ui_dir"
mv "$notebook" "$ui_dir/ui.db"
rmdir "$temp_dir"
trap - EXIT HUP INT TERM
exec duckdb -ui
