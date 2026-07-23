#!/usr/bin/env python3

"""Generate model predictions."""

import argparse
import importlib
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "2d" / "chemical_mapping"
INPUT_FILE = DATA_DIR / "rnagym_2d.parquet"
PSEUDOBASE_FILE = DATA_DIR / "rnagym_pseudobase.parquet"
OUTPUT_DIR = DATA_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment")
    parser.add_argument(
        "--split",
        choices=["train", "test", "pseudobase", "all"],
        default="test",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be non-negative and smaller than --num-shards")
    return args


def load_profiles(split: str) -> pl.DataFrame:
    profiles = []
    if split in {"train", "test", "all"}:
        chemical_mapping = pl.read_parquet(
            INPUT_FILE, columns=["seqID", "sequence", "split"]
        ).rename({"seqID": "uid"})
        if split != "all":
            chemical_mapping = chemical_mapping.filter(pl.col("split") == split)
        profiles.append(chemical_mapping)

    if split in {"pseudobase", "all"}:
        profiles.append(
            pl.read_parquet(
                PSEUDOBASE_FILE, columns=["pseudobase_ids", "sequence"]
            ).select(
                pl.col("pseudobase_ids").alias("uid"),
                "sequence",
                pl.lit("pseudobase").alias("split"),
            )
        )

    return pl.concat(profiles)


def main() -> None:
    args = parse_args()
    # Python module names use underscores, for example rna-fm -> rna_fm
    adapter_name = args.environment.replace("-", "_")
    adapter = importlib.import_module(f"models.{adapter_name}")

    profiles = load_profiles(args.split)
    sequences = (
        profiles.select("sequence")
        # Fold sequences shared by multiple profiles only once
        .unique()
        .with_columns(pl.col("sequence").str.len_chars().alias("length"))
        .sort(["length", "sequence"], descending=[True, False])
        # Assign length-sorted sequences round-robin across shards
        .with_row_count("rank")
        .filter(pl.col("rank") % args.num_shards == args.shard)
    )

    probabilities = [
        adapter.predict(sequence)
        for sequence in tqdm(
            sequences["sequence"],
            desc=f"{args.environment} shard {args.shard + 1}/{args.num_shards}",
        )
    ]
    predictions = sequences.select("sequence").with_columns(
        pl.Series(
            "probabilities",
            probabilities,
            dtype=pl.List(pl.Float32),
        )
    )
    output = (
        profiles.join(predictions, on="sequence")
        .select("uid", "probabilities")
        .sort("uid")
    )
    output_dir = OUTPUT_DIR / args.environment
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.split}-{args.shard}_of_{args.num_shards}.parquet"
    output.write_parquet(output_file)
    print(f"Wrote {output.height:,} profiles to {output_file}")


if __name__ == "__main__":
    main()
