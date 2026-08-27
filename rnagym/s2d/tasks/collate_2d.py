#!/usr/bin/env python3

"""Collate RNAGym chemical mapping and discrete structure data."""

import json
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from rnagym.config import Config2D, Config3D
from rnagym.s3d.curation import monomer_filter
from rnagym.sequences import fitness_sequences, update_registry

from ..models.utils import dot_bracket, pairs_to_dot_bracket, parse_pairs

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


def consolidate_structures(
    rows: list[dict[str, str]], source: str, source_count: int
) -> pl.DataFrame:
    """Collapse exact duplicates while retaining every source identifier."""
    structures = (
        pl.DataFrame(rows)
        .sort("source_id")
        .group_by(["sequence", "secondary_structure"], maintain_order=True)
        .agg(pl.col("source_id").str.join("|").alias("source_id"))
        .with_columns(
            pl.concat_str(pl.lit(f"{source}:"), "source_id").alias("uid"),
            pl.col("sequence")
            .map_elements(lambda sequence: [True] * len(sequence), pl.List(pl.Boolean))
            .alias("resolved"),
        )
        .select("uid", "sequence", "secondary_structure", "resolved")
    )
    print(
        f"{source}: {source_count:,} source -> {len(rows):,} canonical -> "
        f"{structures.height:,} unique"
    )
    return structures


def get_bprna_structures(path: Path = Config2D.BPRNA_FILE) -> pl.DataFrame:
    """Load canonical structures from the official bpRNA-1m archive."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path.name}")

    rows = []
    with ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".dbn"))
        for name in names:
            lines = archive.read(name).decode().splitlines()
            source_id = Path(name).stem
            if len(lines) < 5 or lines[0] != f"#Name: {source_id}":
                raise ValueError(f"Invalid bpRNA record: {name}")
            sequence, structure = (line.strip() for line in lines[-2:])
            sequence = sequence.upper().replace("T", "U")
            if any(base not in "ACGU" for base in sequence):
                continue
            if len(sequence) != len(structure):
                raise ValueError(f"Sequence/structure length mismatch: {name}")
            parse_pairs(structure)
            rows.append(
                {
                    "source_id": source_id,
                    "sequence": sequence,
                    "secondary_structure": structure,
                }
            )
    return consolidate_structures(rows, "bprna", len(names))


def get_efold_challenging_structures(
    paths: tuple[tuple[str, Path], ...] = (
        ("lncrna", Config2D.EFOLD_LNCRNA_FILE),
        ("viral", Config2D.EFOLD_VIRAL_FILE),
    ),
) -> pl.DataFrame:
    """Load eFold's long noncoding RNA and viral challenging sets."""
    missing = [path.name for _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {', '.join(missing)}")

    rows = []
    source_count = 0
    for subset, path in paths:
        with path.open() as handle:
            records = json.load(handle)
        source_count += len(records)
        for source_id, record in sorted(records.items()):
            sequence = record["sequence"].upper().replace("T", "U")
            if any(base not in "ACGU" for base in sequence):
                continue
            rows.append(
                {
                    "source_id": f"{subset}:{source_id}",
                    "sequence": sequence,
                    "secondary_structure": pairs_to_dot_bracket(
                        len(sequence), record["structure"]
                    ),
                }
            )
    return consolidate_structures(rows, "efold_challenging", source_count)


def get_pdb_candidates() -> list[tuple[str, Path, str]]:
    """Select self-structured RNA monomers from the annotated PDB chains."""
    source = pl.read_parquet(Config3D.ANNOTATED_CHAINS_FILE)
    candidates = (
        source.filter(
            monomer_filter()
            & pl.col("sequence").str.contains("^[ACGU]+$")
            & (pl.col("sequence").str.len_chars() <= Config2D.MAX_SEQUENCE_LENGTH)
        )
        .select("pdb_id", "asym_id", "sequence")
        .sort("pdb_id", "asym_id")
    )
    candidates = [
        (
            f"pdb:{pdb_id}_{asym_id}",
            Config3D.chain_file(pdb_id, asym_id),
            sequence,
        )
        for pdb_id, asym_id, sequence in candidates.iter_rows()
    ]
    if len({uid for uid, _, _ in candidates}) != len(candidates):
        raise RuntimeError("PDB chain identifiers must be unique")
    print(f"PDB: {source.height:,} source chains -> {len(candidates):,} monomers")
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
            command = f"{Config2D.MC_ANNOTATE} {pdb_file}"
            subprocess.run(shlex.split(command), stdout=output, check=True)
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


def print_summary(data: pl.DataFrame, path: Path) -> None:
    """Print output row and sequence counts."""
    print(f"Wrote {data.height:,} rows to {path}")
    print(f"Unique sequences: {data['sequence'].n_unique():,}")


def write_parquet(data: pl.DataFrame, path: Path) -> None:
    """Write a Parquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    data.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def main() -> None:
    """Collate and write chemical mapping and discrete structure datasets."""
    filtered = get_source_profiles()
    bprna = get_bprna_structures()
    efold_challenging = get_efold_challenging_structures()
    pseudobase = get_pseudobase_structures()
    pdb = get_pdb_structures()
    datasets = {
        "mapping": filtered,
        "bprna": bprna,
        "efold_challenging": efold_challenging,
        "pseudobase": pseudobase,
        "pdb": pdb,
    }
    datasets = {
        name: data.filter(
            pl.col("sequence").str.len_chars() <= Config2D.MAX_SEQUENCE_LENGTH
        )
        for name, data in datasets.items()
    }
    filtered = datasets["mapping"]
    structures = pl.concat(data for name, data in datasets.items() if name != "mapping")
    modalities = pl.concat(
        [
            *(
                data.select("sequence").with_columns(pl.lit(name).alias("modality"))
                for name, data in datasets.items()
            ),
            fitness_sequences(),
        ]
    )
    if Config3D.TARGET_FILE.is_file():
        modalities = pl.concat(
            [
                modalities,
                pl.read_parquet(Config3D.TARGET_FILE)
                .select("sequence")
                .with_columns(pl.lit("3d").alias("modality")),
            ]
        )
    registry = update_registry(modalities)

    # Count each cluster once per modality and fold
    fold_counts = (
        modalities.join(registry, on="sequence")
        .group_by(["modality", "fold"])
        .agg(pl.col("cluster_rep").n_unique().alias("clusters"))
        .sort(["modality", "fold"])
    )
    print(fold_counts)

    filtered = filtered.join(
        registry.select("sequence_id", "sequence"), on="sequence", how="left"
    ).select("uid", "sequence_id", pl.exclude("uid", "sequence_id"))

    write_parquet(filtered, Config2D.MAPPING_FILE)
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
    write_parquet(structures, Config2D.STRUCTURE_FILE)
    print(f"Wrote {structures.height:,} rows to {Config2D.STRUCTURE_FILE}")

    write_parquet(registry, Config2D.SEQUENCE_FILE)
    print(f"Wrote {registry.height:,} rows to {Config2D.SEQUENCE_FILE}")


if __name__ == "__main__":
    main()
