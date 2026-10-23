#!/usr/bin/env bash

set -euo pipefail

snapshot=$(python -c 'from rnagym.config import Config3D; print(Config3D.CCD_SNAPSHOT)')
archive=$(python -c 'from rnagym.config import Config3D; print(Config3D.CCD_FILE)')
table=$(python -c 'from rnagym.config import Config3D; print(Config3D.CCD_TABLE_FILE)')
# Belongs to the 20260101 snapshot; a new snapshot needs a new checksum
checksum=4a487a587dd146bbfcf0c32de9fb057aef9345df63c15800bc344f7ae5f8c0ae

mkdir -p "$(dirname -- "$archive")"
# Concurrent fetches would otherwise race on the download and the table
exec 9>"$(dirname -- "$archive")/.lock"
flock 9

if ! echo "$checksum  $archive" | sha256sum --check --status 2>/dev/null; then
	temporary=$(mktemp "$archive.XXXXXX")
	trap 'rm -f -- "$temporary"' EXIT
	curl -L --fail --retry 3 --progress-bar --output "$temporary" \
		"https://pdbsnapshots.s3.us-west-2.amazonaws.com/$snapshot/pub/pdb/data/monomers/components.cif.gz"
	echo "$checksum  $temporary" | sha256sum --check --status
	chmod 664 "$temporary"
	mv -- "$temporary" "$archive"
fi

if [[ ! -s $table || $table -ot $archive ]]; then
	python -c 'import sys
from pathlib import Path
from rnagym.s3d.util.polymer import write_ccd_table
write_ccd_table(Path(sys.argv[1]), Path(sys.argv[2]))' "$archive" "$table"
fi

test -s "$table"
