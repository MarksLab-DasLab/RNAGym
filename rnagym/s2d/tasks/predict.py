#!/usr/bin/env python3

"""Generate model prediction shards."""

import argparse
import importlib
from enum import Enum
from itertools import groupby

import polars as pl
from tqdm.auto import tqdm

from rnagym.config import Config2D


class Dataset(Enum):
    """Prediction datasets."""

    MAPPING = "mapping"
    STRUCTURES = "2d"


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
    if dataset is Dataset.MAPPING:
        return pl.read_parquet(
            Config2D.MAPPING_FILE, columns=["sequence_id", "sequence"]
        ).unique()
    if dataset is Dataset.STRUCTURES:
        return pl.read_parquet(
            Config2D.STRUCTURE_FILE, columns=["sequence_id", "sequence"]
        ).unique()
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


def predict_sequences(adapter: object, sequences: pl.Series, description: str) -> list:
    """Predict sequences using equal-length batches when supported."""
    predict_batch = getattr(adapter, "predict_batch", None)
    if predict_batch is None:
        return [
            adapter.predict(sequence) for sequence in tqdm(sequences, desc=description)
        ]

    results = []
    with tqdm(total=len(sequences), desc=description) as progress:
        for length, group in groupby(sequences, key=len):
            group = list(group)
            size = adapter.batch_size(length)
            for start in range(0, len(group), size):
                batch = group[start : start + size]
                results.extend(predict_batch(batch))
                progress.update(len(batch))
    return results


def predict_dataset(
    adapter: object,
    environment: str,
    dataset: Dataset,
    shard: int,
    num_shards: int,
) -> None:
    """Generate and write one dataset shard."""
    output_dir = Config2D.PREDICTION_DIR / environment / dataset.value
    output_file = output_dir / f"{shard}.parquet"
    if output_file.is_file() and output_file.stat().st_size:
        print(f"Skipping {output_file}: already exists")
        return

    profiles = load_profiles(dataset)
    sequences = get_sequences(profiles, num_shards).filter(pl.col("shard") == shard)
    description = f"{environment} {dataset.value} shard {shard + 1}/{num_shards}"
    results = predict_sequences(adapter, sequences["sequence"], description)
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
    output = (
        profiles.join(predictions, on="sequence")
        .select("sequence_id", "probabilities", "structures")
        .sort("sequence_id")
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
    model = args.environment.replace("-", "_")
    adapter = importlib.import_module(f"rnagym.s2d.models.{model}")
    predict_dataset(
        adapter,
        args.environment,
        args.dataset,
        args.shard,
        args.num_shards,
    )


if __name__ == "__main__":
    main()
