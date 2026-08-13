"""Maintain the shared RNAGym sequence registry and similarity clusters."""

import os
import random
import shlex
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from rnagym.config import Config2D, ConfigFitness

SEQUENCE_SCHEMA = pl.Schema({"sequence_id": pl.String, "sequence": pl.String})
REGISTRY_SCHEMA = pl.Schema(
    {**SEQUENCE_SCHEMA, "cluster_rep": pl.String, "fold": pl.UInt8}
)


def fitness_sequences() -> pl.DataFrame:
    """Load the unique ncRNA fitness assay sequences."""
    return (
        pl.read_csv(ConfigFitness.REFERENCE_FILE)
        .filter(pl.col("RNA_TYPE") != "mRNA-coding")
        .select(
            pl.col("RAW_CONSTRUCT_SEQ")
            .str.to_uppercase()
            .str.replace_all("T", "U")
            .alias("sequence")
        )
        .unique()
        .with_columns(pl.lit("fitness").alias("modality"))
    )


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
        schema=SEQUENCE_SCHEMA,
    )
    return pl.concat([registry, new_entries])


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
            f"--min-seq-id {Config2D.MIN_SEQUENCE_IDENTITY} -c {Config2D.MIN_COVERAGE} "
            f"--cov-mode {Config2D.COVERAGE_MODE} --cluster-mode {Config2D.CLUSTER_MODE} "
            f"--threads {Config2D.MMSEQS_THREADS} -v 3"
        )

        # Keep the MMseqs2 progress bars without its full log
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
                    print(line, end="\n" if "100.00%" in line else "\r", flush=True)
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
    # Label each cluster by its modalities, for example mapping+pseudobase
    signatures = (
        modalities.join(assignments, on="sequence_id")
        .group_by("cluster_rep")
        .agg(pl.col("modality").unique().sort().str.join("+").alias("signature"))
        .sort(["signature", "cluster_rep"])
    )
    rng = random.Random(Config2D.RANDOM_SEED)
    rows = []
    # Shuffle and distribute each signature evenly across folds
    for group in signatures.partition_by("signature", maintain_order=True):
        cluster_reps = group["cluster_rep"].to_list()
        rng.shuffle(cluster_reps)
        offset = rng.randrange(Config2D.NUM_FOLDS)
        rows.extend(
            (cluster_rep, (i + offset) % Config2D.NUM_FOLDS)
            for i, cluster_rep in enumerate(cluster_reps)
        )
    return pl.DataFrame(
        rows,
        schema={"cluster_rep": pl.String, "fold": pl.UInt8},
        orient="row",
    )


def update_registry(modalities: pl.DataFrame) -> pl.DataFrame:
    """Add modality sequences, then update shared clusters and folds."""
    modalities = modalities.select("sequence", "modality").unique()
    previous = load_registry(Config2D.SEQUENCE_FILE, required=False)
    registry = add_sequences(
        previous.select(SEQUENCE_SCHEMA.names()), modalities["sequence"]
    )
    modalities = modalities.join(registry, on="sequence").select(
        "sequence_id", "modality"
    )
    if modalities["sequence_id"].n_unique() != registry.height:
        raise RuntimeError("Every registered sequence must belong to a dataset")

    assignments = (
        previous.select("sequence_id", "cluster_rep")
        if registry.height == previous.height
        else cluster_sequences(registry)
    )
    folds = assign_folds(assignments, modalities)
    registry = (
        registry.join(assignments, on="sequence_id")
        .join(folds, on="cluster_rep")
        .sort("sequence_id")
    )
    if registry["fold"].null_count():
        raise RuntimeError("Some sequences did not receive a fold assignment")
    return registry
