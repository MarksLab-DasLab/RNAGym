#!/usr/bin/env python3

"""Annotate each unique RNAGym sequence with Rfam."""

import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "2d" / "chemical_mapping"
INPUT_FILE = DATA_DIR / "rnagym_2d.parquet"
OUTPUT_DIR = DATA_DIR / "rfam-15.1"
RFAM_DIR = Path(os.environ["RNAGYM_DATABASE_DIR"]) / "Rfam-15.1"
NUM_SHARDS = 16


def write_fasta(sequences: pl.DataFrame, shard: int) -> Path:
    """Write one sequence shard in FASTA format."""
    path = OUTPUT_DIR / f"sequences-{shard:02d}.fasta"
    shard_sequences = sequences.filter(pl.col("shard") == shard)
    with path.open("w") as handle:
        handle.writelines(
            f">{sequence_id}\n{sequence}\n"
            for sequence_id, sequence in shard_sequences.select(
                "sequence_id", "sequence"
            ).iter_rows()
        )
    return path


def scan_shard(shard: int, search_space: float, fasta: Path) -> None:
    """Scan one FASTA shard against Rfam."""
    output = OUTPUT_DIR / f"hits-{shard:02d}.tblout"
    temporary = OUTPUT_DIR / f".hits-{shard:02d}.tmp"
    # Official Rfam settings: https://docs.rfam.org/en/latest/genome-annotation.html
    command = (
        f"cmscan -Z {search_space} --cut_ga --rfam --nohmmonly --noali "
        f"--fmt 2 --clanin {RFAM_DIR / 'Rfam.clanin'} --cpu 1 "
        f"-o /dev/null --tblout {temporary} {RFAM_DIR / 'Rfam.cm'} {fasta}"
    )
    subprocess.run(shlex.split(command), check=True)
    temporary.replace(output)
    print(f"Completed Rfam shard {shard + 1}/{NUM_SHARDS}", flush=True)


def main() -> None:
    """Write sequence shards and annotate them with Rfam."""
    sequences = (
        pl.read_parquet(INPUT_FILE, columns=["sequence_id", "sequence"])
        .unique()
        .with_columns(pl.col("sequence").str.len_chars().alias("length"))
        .sort(["length", "sequence_id"], descending=[True, False])
        .with_row_index("rank")
        .with_columns((pl.col("rank") % NUM_SHARDS).alias("shard"))
    )
    if sequences["sequence_id"].n_unique() != sequences.height:
        raise RuntimeError("Each sequence_id must identify exactly one sequence")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fastas = [write_fasta(sequences, shard) for shard in range(NUM_SHARDS)]
    # cmscan searches both strands by default
    search_space = 2 * sequences["length"].sum() / 1_000_000
    print(
        f"Scanning {sequences.height:,} sequences against Rfam 15.1 "
        f"with {NUM_SHARDS} workers",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=NUM_SHARDS) as executor:
        futures = [
            executor.submit(scan_shard, shard, search_space, fasta)
            for shard, fasta in enumerate(fastas)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
