#!/usr/bin/env python3

from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

INPUT_DIR = Path("./rmdb_compiled")
OUTPUT_FILE = Path("./rnagym_ss.parquet")

SOURCES = {
    **{f"RMDB_dataset_{i}.parquet": "in_vitro" for i in range(1, 10)},
    "RMDB_dataset_extra_clean.parquet": "in_vitro",
    "RMDB_dataset_extra_cotrans.parquet": "cotranscriptional",
    "RMDB_dataset_extra_degradation.parquet": "degradation",
    "RMDB_dataset_extra_invivo.parquet": "in_vivo",
}

COLUMNS = [
    "seqID",
    "sequence",
    "modifier",
    "temperature",
    "chemical",
    "reverse_transcriptase",
    "note",
    "reactivity",
    "reactivity_error",
]


def load_source(filename: str, context: str) -> pl.LazyFrame:
    path = INPUT_DIR / filename
    return pl.scan_parquet(path).select(
        *[pl.col(column).cast(pl.Utf8, strict=False) for column in COLUMNS],
        pl.col("SNR").cast(pl.Float64),
        pl.col("reads").cast(pl.Float64),
        pl.lit(filename).alias("source_file"),
        pl.lit(context).alias("context"),
    )


def main() -> None:
    missing = [filename for filename in SOURCES if not (INPUT_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {', '.join(missing)}")

    with tqdm(
        desc="Merging sources",
        unit="rows",
        unit_scale=True,
    ) as progress:

        def track_progress(batch: pl.DataFrame) -> pl.DataFrame:
            progress.update(batch.height)
            return batch

        profiles = pl.concat(
            [load_source(filename, context) for filename, context in SOURCES.items()]
        )

        # Filter by SNR >= 1.0 and take the unique sequence per modifier with the
        # best SNR (tiebroken if necessary by reads)
        filtered = (
            profiles.filter(pl.col("SNR") >= 1.0)
            .map_batches(track_progress, streamable=True)
            .sort(
                ["sequence", "modifier", "SNR", "reads", "seqID", "source_file"],
                descending=[False, False, True, True, False, False],
            )
            .unique(
                subset=["sequence", "modifier"],
                keep="first",
                maintain_order=True,
            )
            .collect(streaming=True)
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(OUTPUT_FILE, compression="zstd", statistics=True)

    print(f"Wrote {filtered.height:,} rows to {OUTPUT_FILE}")
    print(f"Unique sequences: {filtered['sequence'].n_unique():,}")

    counts = filtered.select(
        ((pl.col("context") == "in_vitro") & (pl.col("modifier") == "DMS"))
        .sum()
        .alias("DMS"),
        ((pl.col("context") == "in_vitro") & (pl.col("modifier") == "2A3"))
        .sum()
        .alias("2A3"),
        (pl.col("context") == "cotranscriptional").sum().alias("Cotranscriptional"),
        (pl.col("context") == "degradation").sum().alias("Degradation"),
        (pl.col("context") == "in_vivo").sum().alias("In vivo"),
    ).row(0, named=True)

    print("Counts by type:")
    for label, count in counts.items():
        print(f"  {label}: {count:,}")


if __name__ == "__main__":
    main()
