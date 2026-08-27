#!/usr/bin/env bash

set -euo pipefail

release=$(python -c 'from rnagym.config import Config3D; print(Config3D.RNA3DB_RELEASE)')
root=$(python -c 'from rnagym.config import Config3D; print(Config3D.RNA3DB_DIR)')

download() {
	local name=$1 checksum=$2
	local archive=$root/$name
	if ! echo "$checksum  $archive" | sha256sum --check --status 2>/dev/null; then
		mkdir -p "$root"
		curl -L --fail --retry 3 --progress-bar --output "$archive.tmp" \
			"https://github.com/marcellszi/rna3db/releases/download/$release/$name"
		echo "$checksum  $archive.tmp" | sha256sum --check --status
		mv "$archive.tmp" "$archive"
	fi
}

download rna3db-jsons.tar.gz 820ba611042ef3e4ec1de7d00e739eefcbf1f78df1452252cc002e369babae8b
download rna3db-cmscans.tar.gz 53c323c61aef5b9bb72a88bc3980470a13d0f36d559e1bdaf5582173f0a3b751

parse=$root/jsons/parse.json
table=$root/cmscans/${release%-full-release}.tbl
if [[ ! -s $parse || ! -s $table ]]; then
	mkdir -p "$root/jsons" "$root/cmscans"
	tar -xzf "$root/rna3db-jsons.tar.gz" --strip-components=1 \
		-C "$root/jsons" 'rna3db-jsons/parse.json'
	tar -xzf "$root/rna3db-cmscans.tar.gz" --strip-components=1 \
		-C "$root/cmscans" "rna3db-cmscans/${release%-full-release}.tbl"
fi

test -s "$parse"
test -s "$table"
