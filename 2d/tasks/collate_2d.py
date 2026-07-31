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

from tasks.utils import SEQUENCE_SCHEMA, add_sequences, load_registry

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "2d" / "chemical_mapping"
INPUT_DIR = DATA_DIR / "raw_data"
OUTPUT_FILE = DATA_DIR / "rnagym_map.parquet"
PSEUDOBASE_FILE = INPUT_DIR / "pseudobase.csv"
PSEUDOBASE_OUTPUT_FILE = DATA_DIR / "rnagym_pb.parquet"
SEQUENCE_OUTPUT_FILE = DATA_DIR / "rnagym_sequences.parquet"

MIN_SEQUENCE_IDENTITY = 0.40
MIN_COVERAGE = 0.80
COVERAGE_MODE = 0  # Require coverage of both sequences
CLUSTER_MODE = 0  # Greedy set-cover clustering
NUM_FOLDS = 5
RANDOM_SEED = 42
MMSEQS_THREADS = 1  # Greedy clustering is non-deterministic across threads

SOURCES = {
    **{f"RMDB_dataset_{i}.parquet": "in_vitro" for i in range(1, 10)},
    "RMDB_dataset_extra_clean.parquet": "in_vitro",
    "RMDB_dataset_extra_cotrans.parquet": "cotranscriptional",
    "RMDB_dataset_extra_degradation.parquet": "degradation",
    "RMDB_dataset_extra_invivo.parquet": "in_vivo",
}

CONDITION_COLUMNS = [
    "sequence",
    "modifier",
    "temperature",
    "chemical",
    "reverse_transcriptase",
    "note",
]
REACTIVITY_COLUMNS = ["reactivity", "reactivity_error"]
MEASUREMENT_COLUMNS = [*REACTIVITY_COLUMNS, "SNR", "reads"]
GROUP_COLUMNS = ["experiment_series", *CONDITION_COLUMNS, "context"]


def load_source(filename: str, context: str) -> pl.LazyFrame:
    """Load and normalize one chemical mapping source."""
    path = INPUT_DIR / filename
    return pl.scan_parquet(path).select(
        pl.col("seqID", *CONDITION_COLUMNS).cast(pl.String, strict=False),
        # Raw Parquets store numeric vectors as bracketed comma-separated strings
        pl.col(REACTIVITY_COLUMNS)
        .cast(pl.String)
        .str.strip_chars("[]")
        .str.split(",")
        .cast(pl.List(pl.Float64)),
        pl.col("SNR").cast(pl.Float64),
        pl.col("reads").cast(pl.Int64, strict=False),
        pl.lit(filename).alias("source_file"),
        pl.lit(context).alias("context"),
    )


def get_source_profiles() -> pl.DataFrame:
    """Keep one representative and its repeats per experiment and condition."""
    missing = [filename for filename in SOURCES if not (INPUT_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {', '.join(missing)}")

    with tqdm(desc="Merging sources", unit="rows", unit_scale=True) as progress:

        def track_progress(batch: pl.DataFrame) -> pl.DataFrame:
            """Update the source merge progress bar."""
            progress.update(batch.height)
            return batch

        profiles = pl.concat(
            [load_source(filename, context) for filename, context in SOURCES.items()]
        )

        filtered = (
            profiles.filter(pl.col("SNR") >= 1.0)
            .map_batches(track_progress, streamable=True)
            # Group related RMDB entries such as NAME_DMS_0001.1 and NAME_DMS_0002.1
            .with_columns(
                pl.col("seqID")
                .str.extract(r"^(.*)_\d+\.\d+$", 1)
                .alias("experiment_series")
            )
            .sort(
                ["SNR", "reads", "seqID"],
                descending=[True, True, False],
                nulls_last=True,
            )
            # Remove source rows that point to the exact same measurement
            .unique(
                subset=[*GROUP_COLUMNS, *MEASUREMENT_COLUMNS],
                keep="first",
                maintain_order=True,
            )
            .group_by(GROUP_COLUMNS, maintain_order=True)
            # For [best, repeat1, repeat2], keep best at the top level and
            # store [repeat1, repeat2] in the replicates list
            .agg(
                pl.first("seqID").alias("uid"),
                pl.first(*MEASUREMENT_COLUMNS, "source_file"),
                pl.struct(pl.col("seqID").alias("uid"), *MEASUREMENT_COLUMNS)
                .slice(1)
                .alias("replicates"),
            )
            .drop("experiment_series")
            .sort("uid")
            .collect(engine="streaming")
        )
        repeats = filtered["replicates"].list.len().sum()
        print(
            f"Chemical mapping: {filtered.height:,} representatives + "
            f"{repeats:,} repeats"
        )
        return filtered


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
        pl.col("pseudobase_id").str.join("|").alias("pseudobase_ids")
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
    """Write sequence IDs and sequences in FASTA format."""
    with path.open("w") as handle:
        handle.writelines(
            f">{sequence_id}\n{sequence}\n"
            for sequence_id, sequence in sequences.iter_rows()
        )


def cluster_sequences(sequence_table: pl.DataFrame) -> pl.DataFrame:
    """Cluster sequences with MMseqs2."""
    sequence_table = sequence_table.sort("sequence")

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

        # Run the MMseqs2 command, capturing only progress bars and descriptors
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
            new_columns=["cluster_rep", "sequence_id"],
        )

    if (
        cluster_members.height != sequence_table.height
        or cluster_members["sequence_id"].n_unique() != sequence_table.height
    ):
        raise RuntimeError("MMseqs2 output does not assign every sequence exactly once")

    assignments = sequence_table.join(
        cluster_members, on="sequence_id", how="left"
    ).select("sequence_id", "cluster_rep")
    if assignments["cluster_rep"].null_count():
        raise RuntimeError("Some sequences did not receive a cluster assignment")
    print(f"MMseqs2 clusters: {assignments['cluster_rep'].n_unique():,}")
    return assignments


def assign_folds(assignments: pl.DataFrame, modalities: pl.DataFrame) -> pl.DataFrame:
    """Balance cluster folds across modality signatures."""
    # Label each cluster by its modalities, for example chemical_mapping+pseudobase
    signatures = (
        modalities.join(assignments, on="sequence_id")
        .group_by("cluster_rep")
        .agg(pl.col("modality").unique().sort().str.join("+").alias("signature"))
        .sort(["signature", "cluster_rep"])
    )
    rng = random.Random(RANDOM_SEED)
    rows = []
    # Shuffle and distribute each signature evenly across folds
    for group in signatures.partition_by("signature", maintain_order=True):
        cluster_reps = group["cluster_rep"].to_list()
        rng.shuffle(cluster_reps)
        offset = rng.randrange(NUM_FOLDS)
        rows.extend(
            (cluster_rep, (i + offset) % NUM_FOLDS)
            for i, cluster_rep in enumerate(cluster_reps)
        )
    return pl.DataFrame(
        rows,
        schema={"cluster_rep": pl.String, "fold": pl.UInt8},
        orient="row",
    )


def get_modality_sequences(
    data: pl.DataFrame, registry: pl.DataFrame, modality: str
) -> pl.DataFrame:
    """Map one modality's unique sequences to registry identifiers."""
    return (
        data.select("sequence")
        .unique()
        .join(registry, on="sequence")
        .select("sequence_id", "sequence")
        .with_columns(pl.lit(modality).alias("modality"))
    )


def print_summary(data: pl.DataFrame) -> None:
    """Print output row and sequence counts."""
    print(f"Wrote {data.height:,} rows to {OUTPUT_FILE}")
    print(f"Unique sequences: {data['sequence'].n_unique():,}")


def main() -> None:
    """Collate and write the chemical mapping and PseudoBase datasets."""
    filtered = get_source_profiles()
    pseudobase = get_pseudobase_structures()
    registry = load_registry(SEQUENCE_OUTPUT_FILE, required=False).select(
        SEQUENCE_SCHEMA.names()
    )
    registry = add_sequences(registry, filtered["sequence"])
    registry = add_sequences(registry, pseudobase["sequence"])

    mapping_sequences = get_modality_sequences(filtered, registry, "chemical_mapping")
    pseudobase_sequences = get_modality_sequences(pseudobase, registry, "pseudobase")
    modalities = pl.concat([mapping_sequences, pseudobase_sequences]).select(
        "sequence_id", "modality"
    )
    if modalities["sequence_id"].n_unique() != registry.height:
        raise RuntimeError("Every registered sequence must belong to a dataset")

    assignments = cluster_sequences(registry)
    folds = assign_folds(assignments, modalities)
    registry = (
        registry.join(assignments, on="sequence_id")
        .join(folds, on="cluster_rep")
        .sort("sequence_id")
    )
    if registry["fold"].null_count():
        raise RuntimeError("Some sequences did not receive a fold assignment")

    # Count each cluster once per modality and fold
    fold_counts = (
        modalities.join(
            registry.select("sequence_id", "cluster_rep", "fold"), on="sequence_id"
        )
        .group_by(["modality", "fold"])
        .agg(pl.col("cluster_rep").n_unique().alias("clusters"))
        .sort(["modality", "fold"])
    )
    print(fold_counts)

    filtered = filtered.join(
        mapping_sequences.select("sequence_id", "sequence"),
        on="sequence",
        how="left",
    ).select("uid", "sequence_id", pl.exclude("uid", "sequence_id"))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(OUTPUT_FILE, compression="zstd", statistics=True)
    print_summary(filtered)

    pseudobase = pseudobase.join(
        pseudobase_sequences.select("sequence_id", "sequence"),
        on="sequence",
        how="left",
    ).select(
        "pseudobase_ids",
        "sequence_id",
        "sequence",
        "secondary_structure",
    )
    pseudobase.write_parquet(
        PSEUDOBASE_OUTPUT_FILE, compression="zstd", statistics=True
    )
    print(f"Wrote {pseudobase.height:,} rows to {PSEUDOBASE_OUTPUT_FILE}")

    registry.write_parquet(SEQUENCE_OUTPUT_FILE, compression="zstd", statistics=True)
    print(f"Wrote {registry.height:,} rows to {SEQUENCE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
