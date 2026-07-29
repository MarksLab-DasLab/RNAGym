"""Shared task utilities."""

from pathlib import Path

import polars as pl

REGISTRY_SCHEMA = pl.Schema({"sequence_id": pl.String, "sequence": pl.String})


def load_registry(path: Path, required: bool = True) -> pl.DataFrame:
    """Load and validate the sequence registry."""
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing sequence registry: {path}")
        return pl.DataFrame(schema=REGISTRY_SCHEMA)

    registry = pl.read_parquet(path)
    if registry.schema != REGISTRY_SCHEMA:
        raise RuntimeError(f"Invalid sequence registry schema: {registry.schema}")
    if (
        registry["sequence_id"].n_unique() != registry.height
        or registry["sequence"].n_unique() != registry.height
    ):
        raise RuntimeError("Sequence registry identifiers and sequences must be unique")

    id_numbers = (
        registry["sequence_id"]
        .str.strip_prefix("sequence_")
        .cast(pl.UInt64, strict=False)
    )
    if registry.height and (
        id_numbers.null_count()
        or id_numbers.min() != 0
        or id_numbers.max() != registry.height - 1
    ):
        raise RuntimeError("Sequence registry identifiers must be contiguous")
    return registry


def add_sequences(registry: pl.DataFrame, sequences: pl.Series) -> pl.DataFrame:
    """Append identifiers for sequences not already in the registry."""
    new_sequences = sorted(set(sequences) - set(registry["sequence"]))
    first_id = registry.height
    new_entries = pl.DataFrame(
        {
            "sequence_id": [
                f"sequence_{i:07d}"
                for i in range(first_id, first_id + len(new_sequences))
            ],
            "sequence": new_sequences,
        },
        schema=REGISTRY_SCHEMA,
    )
    return pl.concat([registry, new_entries])
