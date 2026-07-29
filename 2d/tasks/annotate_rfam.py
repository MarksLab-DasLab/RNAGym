#!/usr/bin/env python3

"""Annotate each unique RNAGym sequence with Rfam."""

import os
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from tasks.utils import load_registry

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "2d" / "chemical_mapping"
SEQUENCE_FILE = DATA_DIR / "rnagym_sequences.parquet"
OUTPUT_FILE = DATA_DIR / "rnagym_rfams.parquet"
RFAM_DIR = Path(os.environ["RNAGYM_DATABASE_DIR"]) / "Rfam-15.1"
NUM_SHARDS = 16

HIT_FIELDS = {
    "sequence_id": (3, pl.String),
    "accession": (2, pl.String),
    "name": (1, pl.String),
    "clan": (5, pl.String),
    "e_value": (17, pl.Float64),
    "score": (16, pl.Float64),
    "sequence_start": (9, pl.Int64),
    "sequence_end": (10, pl.Int64),
    "strand": (11, pl.String),
    "truncation": (12, pl.String),
    "overlap": (19, pl.String),
}
HIT_SCHEMA = pl.Schema({name: dtype for name, (_, dtype) in HIT_FIELDS.items()})


def write_fasta(sequences: pl.DataFrame, shard: int, work_dir: Path) -> Path:
    """Write one sequence shard in FASTA format."""
    path = work_dir / f"sequences-{shard:02d}.fasta"
    shard_sequences = sequences.filter(pl.col("shard") == shard)
    with path.open("w") as handle:
        handle.writelines(
            f">{sequence_id}\n{sequence}\n"
            for sequence_id, sequence in shard_sequences.select(
                "sequence_id", "sequence"
            ).iter_rows()
        )
    return path


def scan_shard(shard: int, search_space: float, fasta: Path) -> Path:
    """Scan one FASTA shard against Rfam."""
    output = fasta.with_name(f"hits-{shard:02d}.tblout")
    # Official Rfam settings: https://docs.rfam.org/en/latest/genome-annotation.html
    command = (
        f"cmscan -Z {search_space} --cut_ga --rfam --nohmmonly --noali "
        f"--fmt 2 --clanin {RFAM_DIR / 'Rfam.clanin'} --cpu 1 "
        f"-o /dev/null --tblout {output} {RFAM_DIR / 'Rfam.cm'} {fasta}"
    )
    subprocess.run(shlex.split(command), check=True)
    print(f"Completed Rfam shard {shard + 1}/{NUM_SHARDS}", flush=True)
    return output


def read_hits(paths: list[Path]) -> pl.DataFrame:
    """Read Rfam hits from Infernal format-2 tabular output."""
    hits = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.split(maxsplit=29)
                if len(fields) < 29:
                    raise RuntimeError(f"Malformed Rfam hit: {line.rstrip()}")
                hit = {name: fields[index] for name, (index, _) in HIT_FIELDS.items()}
                hit["clan"] = None if hit["clan"] == "-" else hit["clan"]
                hits.append(hit)
    return pl.DataFrame(hits, schema=HIT_SCHEMA)


def write_annotations(sequences: pl.DataFrame, hits: pl.DataFrame) -> None:
    """Write one nested Rfam hit list per sequence."""
    hit_columns = [column for column in HIT_SCHEMA.names() if column != "sequence_id"]
    grouped_hits = (
        hits.sort(["sequence_id", "e_value", "accession"])
        .group_by("sequence_id", maintain_order=True)
        .agg(pl.struct(hit_columns).alias("rfam_hits"))
    )
    hit_list_type = grouped_hits.schema["rfam_hits"]
    annotations = (
        sequences.select("sequence_id")
        .join(grouped_hits, on="sequence_id", how="left")
        .with_columns(pl.col("rfam_hits").fill_null(pl.lit([], dtype=hit_list_type)))
        .sort("sequence_id")
    )
    temporary = OUTPUT_FILE.with_suffix(".tmp")
    annotations.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(OUTPUT_FILE)
    print(
        f"Wrote {annotations.height:,} sequences and {hits.height:,} hits to {OUTPUT_FILE}"
    )


def main() -> None:
    """Scan every registered sequence and write compact Rfam annotations."""
    sequences = (
        load_registry(SEQUENCE_FILE)
        .with_columns(pl.col("sequence").str.len_chars().alias("length"))
        .sort(["length", "sequence_id"], descending=[True, False])
        .with_row_index("rank")
        .with_columns((pl.col("rank") % NUM_SHARDS).alias("shard"))
    )
    search_space = 2 * sequences["length"].sum() / 1_000_000
    print(
        f"Scanning {sequences.height:,} sequences against Rfam 15.1 "
        f"with {NUM_SHARDS} workers",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="rnagym-rfam-") as temporary_dir:
        work_dir = Path(temporary_dir)
        fastas = [
            write_fasta(sequences, shard, work_dir) for shard in range(NUM_SHARDS)
        ]
        with ThreadPoolExecutor(max_workers=NUM_SHARDS) as executor:
            futures = [
                executor.submit(scan_shard, shard, search_space, fasta)
                for shard, fasta in enumerate(fastas)
            ]
            outputs = [future.result() for future in futures]
        hits = read_hits(outputs)

    write_annotations(sequences, hits)


if __name__ == "__main__":
    main()
