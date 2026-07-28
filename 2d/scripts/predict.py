#!/usr/bin/env python3

"""Generate model predictions."""

import argparse
import importlib
from enum import Enum
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "2d" / "chemical_mapping"
INPUT_FILE = DATA_DIR / "rnagym_2d.parquet"
PSEUDOBASE_FILE = DATA_DIR / "rnagym_pseudobase.parquet"
OUTPUT_DIR = DATA_DIR / "predictions"


class Dataset(Enum):
    """Prediction datasets."""

    CHEMICAL_MAPPING = "chemical_mapping"
    PSEUDOBASE = "pseudobase"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("environment")
    parser.add_argument("dataset", type=Dataset)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be non-negative and smaller than --num-shards")
    return args


def load_profiles(dataset: Dataset) -> pl.DataFrame:
    """Load profiles for one dataset."""
    if dataset is Dataset.CHEMICAL_MAPPING:
        return pl.read_parquet(INPUT_FILE, columns=["sequence_id", "sequence"]).unique()
    if dataset is Dataset.PSEUDOBASE:
        return pl.read_parquet(
            PSEUDOBASE_FILE, columns=["pseudobase_ids", "sequence"]
        ).rename({"pseudobase_ids": "uid"})
    raise ValueError(f"Unknown dataset: {dataset}")


def get_sequences(profiles: pl.DataFrame, num_shards: int) -> pl.DataFrame:
    """Assign each unique sequence to a shard."""
    return (
        profiles.select("sequence")
        .unique()
        .with_columns(pl.col("sequence").str.len_chars().alias("length"))
        .sort(["length", "sequence"], descending=[True, False])
        .with_row_index("rank")
        .with_columns((pl.col("rank") % num_shards).alias("shard"))
    )


def predict_dataset(
    adapter: object,
    environment: str,
    dataset: Dataset,
    shard: int,
    num_shards: int,
) -> None:
    """Generate and write one dataset shard."""
    output_dir = OUTPUT_DIR / environment / dataset.value
    output_file = output_dir / f"{shard}.parquet"
    if output_file.is_file() and output_file.stat().st_size:
        print(f"Skipping {output_file}: already exists")
        return

    profiles = load_profiles(dataset)
    sequences = get_sequences(profiles, num_shards).filter(pl.col("shard") == shard)
    results = [
        adapter.predict(sequence)
        for sequence in tqdm(
            sequences["sequence"],
            desc=f"{environment} {dataset.value} shard {shard + 1}/{num_shards}",
        )
    ]
    predictions = sequences.select("sequence").with_columns(
        pl.Series(
            "probabilities",
            [result["probabilities"] for result in results],
            dtype=pl.List(pl.Float32),
        ),
        pl.Series(
            "structures",
            [result["structures"] for result in results],
            dtype=pl.List(
                pl.Struct(
                    {
                        "method": pl.Utf8,
                        "dot_bracket": pl.Utf8,
                    }
                )
            ),
        ),
    )
    if dataset is Dataset.CHEMICAL_MAPPING:
        output = (
            profiles.join(predictions, on="sequence")
            .select("sequence_id", "probabilities", "structures")
            .sort("sequence_id")
        )
    else:
        output = (
            profiles.join(predictions, on="sequence")
            .select("uid", "probabilities", "structures")
            .sort("uid")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_file = output_dir / f".{shard}.tmp"
    output.write_parquet(temporary_file)
    temporary_file.replace(output_file)
    print(f"Wrote {output.height:,} predictions to {output_file}")


def main() -> None:
    """Generate model prediction shards."""
    args = parse_args()
    # Python module names use underscores, for example rna-fm -> rna_fm
    adapter = importlib.import_module(f"models.{args.environment.replace('-', '_')}")
    predict_dataset(
        adapter,
        args.environment,
        args.dataset,
        args.shard,
        args.num_shards,
    )


if __name__ == "__main__":
    main()
