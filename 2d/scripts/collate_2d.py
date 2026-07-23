#!/usr/bin/env python3

"""Collate the RNAGym chemical mapping and PseudoBase data."""

import os
import random
import shlex
import subprocess
import tempfile
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "2d" / "chemical_mapping"
INPUT_DIR = DATA_DIR / "raw_data"
OUTPUT_FILE = DATA_DIR / "rnagym_2d.parquet"
PSEUDOBASE_FILE = INPUT_DIR / "pseudobase.csv"
PSEUDOBASE_OUTPUT_FILE = DATA_DIR / "rnagym_pseudobase.parquet"

MIN_SEQUENCE_IDENTITY = 0.40
MIN_COVERAGE = 0.80
COVERAGE_MODE = 0  # Require coverage of both sequences
CLUSTER_MODE = 0  # Greedy set-cover clustering
TRAIN_FRACTION = 0.80
RANDOM_SEED = 42
MMSEQS_THREADS = 1  # Greedy clustering is non-deterministic across threads

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


def get_source_profiles() -> pl.DataFrame:
    """
    Filter by SNR >= 1.0 and take the unique sequence per modifier with the
    best SNR (tiebroken if necessary by reads)
    """
    missing = [filename for filename in SOURCES if not (INPUT_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {', '.join(missing)}")

    with tqdm(desc="Merging sources", unit="rows", unit_scale=True) as progress:

        def track_progress(batch: pl.DataFrame) -> pl.DataFrame:
            progress.update(batch.height)
            return batch

        profiles = pl.concat(
            [load_source(filename, context) for filename, context in SOURCES.items()]
        )

        return (
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


def get_pseudobase_structures() -> pl.DataFrame:
    """Filter and deduplicate the PseudoBase source entries."""
    if not PSEUDOBASE_FILE.is_file():
        raise FileNotFoundError(f"Missing source file: {PSEUDOBASE_FILE.name}")

    source = pl.read_csv(PSEUDOBASE_FILE).with_columns(
        # PseudoBase uses ":" instead of "." for unpaired bases
        pl.col("bracket_view").str.replace_all(":", ".").alias("secondary_structure")
    )
    filtered = source.filter(
        # Keep continuous structures with canonical RNA and valid pair symbols
        (pl.col("continuous") == "Yes")
        & pl.col("sequence").str.contains(r"^[ACGU]+$")
        & pl.col("secondary_structure").str.contains(r"^[.()\[\]\{\}]+$")
    )
    structures = filtered.group_by(
        ["sequence", "secondary_structure"], maintain_order=True
    ).agg(
        # Retain every ID when multiple entries have the same sequence and structure
        pl.col("pseudobase_id").str.concat("|").alias("pseudobase_ids")
    )

    print(
        f"PseudoBase: {source.height} source -> {filtered.height} filtered -> "
        f"{structures.height} unique"
    )
    return structures.select(
        "pseudobase_ids",
        "sequence",
        "secondary_structure",
    )


def write_fasta(sequences: pl.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        handle.writelines(
            f">{sequence_id}\n{sequence}\n"
            for sequence_id, sequence in sequences.iter_rows()
        )


def cluster_sequences(sequences: pl.Series) -> pl.DataFrame:
    """Cluster sequences with MMseqs2, then split clusters 80/20 into train/test."""
    sequence_values = sorted(sequences.unique().to_list())
    sequence_table = pl.DataFrame(
        {
            "sequence_id": [f"sequence_{i:07d}" for i in range(len(sequence_values))],
            "sequence": sequence_values,
        }
    )

    with tempfile.TemporaryDirectory(prefix="rnagym-mmseqs-") as temporary_dir:
        work_dir = Path(temporary_dir)
        fasta_path = work_dir / "sequences.fasta"
        cluster_prefix = work_dir / "clusters"
        mmseqs_tmp = work_dir / "tmp"

        print(f"Clustering {sequence_table.height:,} unique sequences with MMseqs2")
        write_fasta(sequence_table, fasta_path)

        command = (
            f"mmseqs easy-cluster {fasta_path} {cluster_prefix} {mmseqs_tmp} "
            f"--min-seq-id {MIN_SEQUENCE_IDENTITY} -c {MIN_COVERAGE} "
            f"--cov-mode {COVERAGE_MODE} --cluster-mode {CLUSTER_MODE} "
            f"--threads {MMSEQS_THREADS} -v 3"
        )

        # Run the MMseqs2 command, capturing only progress bars and descriptors.
        # Nonfatal set-cover errors are expected: https://github.com/soedinglab/MMseqs2/issues/765
        with subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "TTY": "1"},
        ) as process:
            description = ""
            for line in process.stdout:
                line = line.rstrip()
                if line.startswith("[") and "%" in line:
                    if "] 0.00%" in line:
                        print(description)
                    print(
                        line,
                        end="\n" if "100.00%" in line else "\r",
                        flush=True,
                    )
                elif line:
                    description = line.split()[0] if f" {work_dir}/" in line else line
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args)

        cluster_members = pl.read_csv(
            cluster_prefix.with_name(f"{cluster_prefix.name}_cluster.tsv"),
            separator="\t",
            has_header=False,
            new_columns=["cluster_id", "sequence_id"],
        )

    if (
        cluster_members.height != sequence_table.height
        or cluster_members["sequence_id"].n_unique() != sequence_table.height
    ):
        raise RuntimeError("MMseqs2 output does not assign every sequence exactly once")

    cluster_ids = sorted(cluster_members["cluster_id"].unique().to_list())
    random.Random(RANDOM_SEED).shuffle(cluster_ids)
    train_cluster_count = round(TRAIN_FRACTION * len(cluster_ids))
    train_clusters = set(cluster_ids[:train_cluster_count])

    cluster_splits = pl.DataFrame(
        {
            "cluster_id": cluster_ids,
            "split": [
                "train" if cluster_id in train_clusters else "test"
                for cluster_id in cluster_ids
            ],
        }
    )

    assignments = (
        sequence_table.join(cluster_members, on="sequence_id", how="left")
        .join(cluster_splits, on="cluster_id", how="left")
        .select("sequence", "split")
    )

    if assignments["split"].null_count() != 0:
        raise RuntimeError("Some sequences did not receive a train/test assignment")

    print(f"MMseqs2 clusters: {len(cluster_ids):,}")
    print(f"Train clusters: {train_cluster_count:,}")
    print(f"Test clusters: {len(cluster_ids) - train_cluster_count:,}")
    return assignments


def print_summary(data: pl.DataFrame) -> None:
    print(f"Wrote {data.height:,} rows to {OUTPUT_FILE}")
    print(f"Unique sequences: {data['sequence'].n_unique():,}")

    counts = data.group_by("split").agg(
        pl.count().alias("rows"),
        pl.col("sequence").n_unique().alias("sequences"),
    )
    print(counts.sort("split"))


def main() -> None:
    filtered = get_source_profiles()
    assignments = cluster_sequences(filtered["sequence"])
    filtered = filtered.join(assignments, on="sequence", how="left")

    if filtered["split"].null_count() != 0:
        raise RuntimeError("Some profiles did not receive a train/test assignment")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(OUTPUT_FILE, compression="zstd", statistics=True)
    print_summary(filtered)

    pseudobase = get_pseudobase_structures()
    pseudobase.write_parquet(
        PSEUDOBASE_OUTPUT_FILE, compression="zstd", statistics=True
    )
    print(f"Wrote {pseudobase.height:,} rows to {PSEUDOBASE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
