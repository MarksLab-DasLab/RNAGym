#!/usr/bin/env bash

checkpoint_root=${RNAGYM_CHECKPOINT_DIR:-/n/lw_groups/marks/ckpt}
umask 002
mkdir -p "$checkpoint_root"
exec 9>"$checkpoint_root/.download.lock"
flock 9

download() {
	local url=$1 destination=$2 checksum=$3 temporary

	mkdir -p "$(dirname -- "$destination")"
	if ! printf '%s  %s\n' "$checksum" "$destination" | sha256sum --check --status 2>/dev/null; then
		temporary=$(mktemp "${destination}.XXXXXX")
		if ! curl -L --fail --retry 3 --output "$temporary" "$url"; then
			rm -f -- "$temporary"
			return 1
		fi
		if ! printf '%s  %s\n' "$checksum" "$temporary" | sha256sum --check --status; then
			rm -f -- "$temporary"
			return 1
		fi
		chmod 664 "$temporary"
		mv -- "$temporary" "$destination"
	fi
}
