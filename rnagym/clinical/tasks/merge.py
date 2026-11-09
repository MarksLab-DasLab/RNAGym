"""Merge one model's shards for one staged file, refusing anything incomplete.

A merge succeeds only if every shard ``0..N-1`` is present exactly once, every
manifest carries the same ``spec_hash``, each shard's parquet matches the
checksum in its manifest, the shards were scored from this staged file, and the
merged row ids equal the staged rows (or the listed row ids) exactly, with no
duplicates. Shards live in ``<score dir>/<name>/<model>``, and every shard's
record must name that model. The merged scores go to
``<merged>/<name>/<model>.parquet``; without ``--scores`` every directory of the
score directory is merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from rnagym.clinical import common
from rnagym.config import ConfigClinical

_SHARD = re.compile(r"shard_(\d{4})_of_(\d{4})\.parquet")


def merge(
    model_dir: Path,
    staged: Path,
    name: str,
    row_ids: Path | None,
    model: str | None = None,
) -> pl.DataFrame:
    """Return one model's merged scores for one staged file.

    With ``model``, every shard must have been scored by that model.
    """
    files = sorted(model_dir.glob("shard_*_of_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no shards in {model_dir}")
    parsed = [_SHARD.fullmatch(f.name) for f in files]
    matches = [m for m in parsed if m is not None]
    if len(matches) != len(files):
        raise ValueError("unexpected shard file names")
    totals = {int(m.group(2)) for m in matches}
    if len(totals) != 1:
        raise ValueError(
            f"shards from runs with different shard counts: {sorted(totals)}"
        )
    num_shards = totals.pop()
    shards = sorted(int(m.group(1)) for m in matches)
    if shards != list(range(num_shards)):
        missing = sorted(set(range(num_shards)) - set(shards))
        raise ValueError(f"missing or duplicated shards; missing {missing[:10]}")

    staged_digest = common.staged_sha256(staged, name)
    expected_rows_file = str(row_ids) if row_ids else None
    frames: list[pl.DataFrame] = []
    hashes: set[str] = set()
    for f, m in zip(files, matches):
        manifest = json.loads(f.with_suffix(".json").read_text())
        spec = manifest["spec"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != manifest["output_sha256"]:
            raise ValueError(f"{f} does not match its manifest checksum")
        if common.spec_hash(spec) != manifest["spec_hash"]:
            raise ValueError(f"{f} manifest spec does not hash to its spec_hash")
        if manifest["shard"] != int(m.group(1)) or spec["num_shards"] != num_shards:
            raise ValueError(f"{f} manifest disagrees with its file name")
        if model is not None and spec["model"] != model:
            raise ValueError(f"{f} was scored by {spec['model']}, not {model}")
        if spec["name"] != name or spec["staged_sha256"] != staged_digest:
            raise ValueError(
                f"{f} was scored from {spec['name']} {spec['staged_sha256'][:12]}, "
                "not this staged file"
            )
        if spec["row_ids_file"] != expected_rows_file:
            raise ValueError(
                f"{f} was scored from row list {spec['row_ids_file']}, "
                f"expected {expected_rows_file}"
            )
        hashes.add(manifest["spec_hash"])
        frames.append(pl.read_parquet(f))
    if len(hashes) != 1:
        raise ValueError(f"shards from different scoring specs: {sorted(hashes)}")

    merged = pl.concat(frames, how="vertical_relaxed").sort("row_id")
    ids = merged["row_id"].to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate row ids across shards")
    expected = common.expected_row_ids(staged, name, row_ids)
    if not np.array_equal(ids, expected):
        raise ValueError(
            f"rows do not match the staged set: {len(ids)} merged, "
            f"{len(expected)} expected"
        )
    contained = merged["contained"].to_numpy()
    if not np.isfinite(merged["score"].to_numpy()[contained]).all():
        raise ValueError("a contained row has no finite score")
    return merged.with_columns(pl.lit(hashes.pop()).alias("spec_hash"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scores",
        type=Path,
        help="One <score dir>/<name>/<model>; without it, every one under --score-dir",
    )
    parser.add_argument("--score-dir", type=Path, default=ConfigClinical.SCORE_DIR)
    parser.add_argument("--staged", type=Path, default=ConfigClinical.STAGED_DIR)
    parser.add_argument("--row-ids", type=Path, help="Row id list the shards scored")
    parser.add_argument("--merged", type=Path, default=ConfigClinical.MERGED_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Merge one scores directory, or every one under the score directory."""
    args = parse_args(argv)
    if args.scores is not None:
        model_dirs = [args.scores]
    else:
        model_dirs = sorted(p for p in args.score_dir.glob("*/*") if p.is_dir())
        if not model_dirs:
            raise FileNotFoundError(
                f"no <name>/<model> directories in {args.score_dir}"
            )
    for model_dir in model_dirs:
        name, model = model_dir.parent.name, model_dir.name
        merged = merge(model_dir, args.staged, name, args.row_ids, model)
        out = args.merged / name / f"{model}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(out)
        print(f"{model_dir}: {len(merged)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
