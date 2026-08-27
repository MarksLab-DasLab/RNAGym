"""Select the post-cutoff 3D benchmark targets."""

from pathlib import Path

import polars as pl

from rnagym.config import Config2D, Config3D
from rnagym.s3d.curation import monomer_filter, quality_filter
from rnagym.s3d.models import BASELINES, homology_columns
from rnagym.sequences import fitness_sequences, update_registry

IDENTIFIERS = ["pdb_id", "asym_id"]
HOMOLOGY_COLUMNS = [column for model in BASELINES for column in homology_columns(model)]


def _add_cached_homology(targets: pl.DataFrame) -> pl.DataFrame:
    """Restore available target-to-training annotations by PDB chain."""
    paths = [Config3D.TARGET_FILE]
    paths.extend(sorted(Config3D.USALIGN_ANNOTATION_DIR.glob("*.parquet")))
    cached = []
    for path in paths:
        if not path.is_file():
            continue
        data = pl.read_parquet(path)
        if not set(IDENTIFIERS).issubset(data.columns):
            continue
        available = [column for column in HOMOLOGY_COLUMNS if column in data.columns]
        if not available:
            continue
        columns = [*IDENTIFIERS, *available]
        cached.append(data.select(columns))
    if not cached:
        return targets

    annotations = (
        pl.concat(cached, how="diagonal_relaxed")
        .group_by(IDENTIFIERS)
        .agg(pl.col(column).drop_nulls().last() for column in HOMOLOGY_COLUMNS)
    )
    targets = targets.join(annotations, on=IDENTIFIERS, how="left", suffix="_cached")
    return targets.with_columns(
        pl.coalesce(f"{column}_cached", column).alias(column)
        for column in HOMOLOGY_COLUMNS
    ).drop([f"{column}_cached" for column in HOMOLOGY_COLUMNS])


def _homology_fields() -> list[pl.Expr]:
    """Return empty typed structural-homology fields."""
    return [
        pl.lit(
            None, dtype=pl.Float64 if column.endswith("_score") else pl.String
        ).alias(column)
        for column in HOMOLOGY_COLUMNS
    ]


def _print_summary(data: pl.DataFrame) -> None:
    """Print a concise target summary."""
    rfams = data.filter(pl.col("rfam_e_value") < 1)["rfam"].n_unique()
    resolution = data["resolution"].mean()
    print(f"Rfam families: {rfams:,}")
    print(f"Average length: {data['length'].mean():.0f} nt")
    print(f"Average measured resolution: {resolution:.2f} Å")
    print(data.group_by("type").len().sort("type"))


def get_split_candidates() -> pl.DataFrame:
    """Select every quality-filtered post-cutoff monomer and multimer."""
    annotated = pl.read_parquet(Config3D.ANNOTATED_CHAINS_FILE)
    quality = annotated.filter(quality_filter())
    candidates = quality.filter(
        (pl.col("published") > Config3D.TARGET_CUTOFF)
        & pl.col("length").is_between(Config3D.MIN_LENGTH, Config3D.MAX_SEQUENCE_LENGTH)
    )
    monomers = candidates.filter(monomer_filter()).with_columns(
        pl.lit("monomer").alias("type")
    )
    multimers = candidates.filter(
        (pl.col("polymer_coverage") > Config3D.MAX_POLYMER_COVERAGE)
        & (pl.col("num_polymer_residues") <= Config3D.MAX_COMPLEX_LENGTH)
    ).with_columns(pl.lit("multimer").alias("type"))
    print(f"All RNAs: {annotated.height:,}")
    print(f"Quality RNAs: {quality.height:,}")
    print(f"After cutoff RNAs: {candidates.height:,}")
    return (
        pl.concat([monomers, multimers])
        .with_columns(_homology_fields())
        .sort("type", "resolution", "length", "pdb_id", "asym_id", nulls_last=True)
    )


def _write_parquet(data: pl.DataFrame, path: Path) -> None:
    """Write one Parquet file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    data.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def main() -> None:
    """Write selected targets and update the shared sequence registry."""
    targets = get_split_candidates()
    structures = pl.read_parquet(Config2D.STRUCTURE_FILE).select("uid", "sequence")
    modalities = pl.concat(
        [
            pl.read_parquet(Config2D.MAPPING_FILE)
            .select("sequence")
            .with_columns(pl.lit("mapping").alias("modality")),
            structures.select(
                "sequence",
                pl.col("uid").str.split(":").list.first().alias("modality"),
            ),
            targets.select("sequence").with_columns(pl.lit("3d").alias("modality")),
            fitness_sequences(),
        ]
    )
    registry = update_registry(modalities)
    targets = (
        targets.join(registry.select("sequence_id", "sequence"), on="sequence")
        .select("type", "sequence_id", pl.exclude("type", "sequence_id"))
        .pipe(_add_cached_homology)
    )

    _print_summary(targets)
    _write_parquet(registry, Config3D.SEQUENCE_FILE)
    _write_parquet(targets, Config3D.TARGET_FILE)
    print(f"Wrote {registry.height:,} sequences to {Config3D.SEQUENCE_FILE}")
    print(f"Wrote {targets.height:,} targets to {Config3D.TARGET_FILE}")


if __name__ == "__main__":
    main()
