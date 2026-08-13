#!/usr/bin/env python3

"""Collate RNAGym chemical mapping and discrete structure data."""

import os
import random
import re
import shlex
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from rnagym.config import Config2D, Config3D

from ..models.utils import dot_bracket
from .utils import SEQUENCE_SCHEMA, add_sequences, load_registry

CWW_PATTERN = re.compile(
    r"^A(\d+)-A(\d+) : \w+-\w+ Ww/Ww.*pairing "
    r"(?:parallel|antiparallel) cis"
)

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
    path = Config2D.RAW_DIR / filename
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
    missing = [
        filename for filename in SOURCES if not (Config2D.RAW_DIR / filename).is_file()
    ]
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
    if not Config2D.PSEUDOBASE_FILE.is_file():
        raise FileNotFoundError(f"Missing source file: {Config2D.PSEUDOBASE_FILE.name}")

    source = pl.read_csv(Config2D.PSEUDOBASE_FILE).with_columns(
        # PseudoBase uses ":" instead of "." for unpaired bases
        pl.col("bracket_view").str.replace_all(":", ".").alias("secondary_structure")
    )
    filtered = source.filter(
        # Keep continuous structures with canonical RNA and valid pair symbols
        (pl.col("continuous") == "Yes")
        & pl.col("sequence").str.contains(r"^[ACGU]+$")
        & pl.col("secondary_structure").str.contains(r"^[.()\[\]\{\}]+$")
    )
    structures = (
        filtered.group_by(["sequence", "secondary_structure"], maintain_order=True)
        .agg(
            # Retain every ID when multiple entries have the same sequence and structure
            pl.col("pseudobase_id").str.join("|").alias("source_id")
        )
        .with_columns(
            pl.concat_str(pl.lit("pseudobase:"), "source_id").alias("uid"),
        )
    )

    print(
        f"PseudoBase: {source.height} source -> {filtered.height} filtered -> "
        f"{structures.height} unique"
    )
    return structures.select(
        "uid",
        "sequence",
        "secondary_structure",
        pl.col("sequence")
        .map_elements(lambda sequence: [True] * len(sequence), pl.List(pl.Boolean))
        .alias("resolved"),
    )


def is_pdb_candidate(row: dict[str, str]) -> bool:
    """Apply the 3D monomer and quality filters to one PDB chain."""
    return Config3D.is_monomer(row) and all(
        base in "ACGU" for base in row["Sequence (unmod.)"]
    )


def get_pdb_candidates() -> list[tuple[str, Path, str]]:
    """Select self-structured RNA monomers from the annotated PDB chains."""
    source = pl.read_csv(
        Config3D.ANNOTATED_CHAINS_FILE, infer_schema_length=None
    ).to_dicts()
    candidates = []
    for row in source:
        if not is_pdb_candidate(row):
            continue
        pdb_id = row["PDB ID"].lower()
        asym_id = row["Asym. Chain ID"]
        candidates.append(
            (
                f"pdb:{pdb_id}_{asym_id}",
                Config3D.CACHE_DIR / pdb_id / asym_id / f"{asym_id}.pdb",
                row["Sequence (unmod.)"],
            )
        )
    candidates.sort()
    if len({uid for uid, _, _ in candidates}) != len(candidates):
        raise RuntimeError("PDB chain identifiers must be unique")
    print(f"PDB: {len(source):,} source chains -> {len(candidates):,} monomers")
    return candidates


def annotate_pdb(candidate: tuple[str, Path, str]) -> dict[str, object]:
    """Convert one cached PDB chain to resolved positions and cWW pairs."""
    uid, pdb_file, sequence = candidate
    if not pdb_file.is_file():
        raise FileNotFoundError(f"Missing cached PDB chain: {pdb_file}")

    annotation_file = pdb_file.with_suffix(".pdb.mcout")
    if not annotation_file.is_file():
        # Cache only complete MC-Annotate output
        temporary = annotation_file.with_name(f"{annotation_file.name}.tmp")
        with temporary.open("w") as output:
            subprocess.run([Config2D.MC_ANNOTATE, pdb_file], stdout=output, check=True)
        temporary.replace(annotation_file)

    positions = {
        int(line[22:26])
        for line in pdb_file.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    }
    if not positions or min(positions) < 1 or max(positions) > len(sequence):
        raise RuntimeError(f"Invalid residue numbering in {pdb_file}")

    pairs = []
    for line in annotation_file.read_text().splitlines():
        match = CWW_PATTERN.match(line)
        if match:
            pairs.append(tuple(sorted((int(match[1]) - 1, int(match[2]) - 1))))

    partner_counts = Counter(position for pair in pairs for position in pair)
    if any(position + 1 not in positions for pair in pairs for position in pair):
        raise RuntimeError(f"MC-Annotate paired an unresolved residue in {pdb_file}")
    ambiguous = {position for position, count in partner_counts.items() if count > 1}
    conflicts = {
        position for pair in pairs if ambiguous.intersection(pair) for position in pair
    }
    pairs = [pair for pair in pairs if not conflicts.intersection(pair)]

    contacts = np.zeros((len(sequence), len(sequence)), dtype=bool)
    for left, right in pairs:
        contacts[left, right] = True

    return {
        "uid": uid,
        "sequence": sequence,
        "secondary_structure": dot_bracket(contacts),
        "resolved": [
            position in positions and position - 1 not in conflicts
            for position in range(1, len(sequence) + 1)
        ],
    }


def get_pdb_structures() -> pl.DataFrame:
    """Extract PDB secondary structures with RNA-Puzzles MC-Annotate."""
    if not Config2D.MC_ANNOTATE.is_file():
        raise FileNotFoundError(f"Missing MC-Annotate: {Config2D.MC_ANNOTATE}")
    return pl.DataFrame(
        annotate_pdb(candidate)
        for candidate in tqdm(get_pdb_candidates(), desc="Annotating PDB chains")
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
            f"--min-seq-id {Config2D.MIN_SEQUENCE_IDENTITY} -c {Config2D.MIN_COVERAGE} "
            f"--cov-mode {Config2D.COVERAGE_MODE} --cluster-mode {Config2D.CLUSTER_MODE} "
            f"--threads {Config2D.MMSEQS_THREADS} -v 3"
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


def print_summary(data: pl.DataFrame, path: Path) -> None:
    """Print output row and sequence counts."""
    print(f"Wrote {data.height:,} rows to {path}")
    print(f"Unique sequences: {data['sequence'].n_unique():,}")


def main() -> None:
    """Collate and write chemical mapping and discrete structure datasets."""
    filtered = get_source_profiles()
    pseudobase = get_pseudobase_structures()
    pdb = get_pdb_structures()
    structures = pl.concat([pseudobase, pdb])
    registry = load_registry(Config2D.SEQUENCE_FILE, required=False).select(
        SEQUENCE_SCHEMA.names()
    )
    registry = add_sequences(registry, filtered["sequence"])
    registry = add_sequences(registry, structures["sequence"])

    datasets = {"mapping": filtered, "pseudobase": pseudobase, "pdb": pdb}
    sequence_tables = {
        name: get_modality_sequences(data, registry, name)
        for name, data in datasets.items()
    }
    modalities = pl.concat(sequence_tables.values()).select("sequence_id", "modality")
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
        sequence_tables["mapping"].select("sequence_id", "sequence"),
        on="sequence",
        how="left",
    ).select("uid", "sequence_id", pl.exclude("uid", "sequence_id"))

    Config2D.MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(Config2D.MAPPING_FILE, compression="zstd", statistics=True)
    print_summary(filtered, Config2D.MAPPING_FILE)

    structures = structures.join(
        registry.select("sequence_id", "sequence"),
        on="sequence",
        how="left",
    ).select(
        "uid",
        "sequence_id",
        "sequence",
        "secondary_structure",
        "resolved",
    )
    structures.write_parquet(
        Config2D.STRUCTURE_FILE, compression="zstd", statistics=True
    )
    print(f"Wrote {structures.height:,} rows to {Config2D.STRUCTURE_FILE}")

    registry.write_parquet(Config2D.SEQUENCE_FILE, compression="zstd", statistics=True)
    print(f"Wrote {registry.height:,} rows to {Config2D.SEQUENCE_FILE}")


if __name__ == "__main__":
    main()
