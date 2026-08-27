"""Shared 3D structure curation filters."""

import polars as pl

from rnagym.config import Config3D


def quality_filter() -> pl.Expr:
    """Select chains meeting the shared experimental quality thresholds."""
    requires_resolution = pl.col("method").str.contains(
        "X-RAY DIFFRACTION|ELECTRON MICROSCOPY"
    )
    resolution_ok = (~requires_resolution) | pl.col("resolution").le(
        Config3D.MAX_RESOLUTION
    ).fill_null(False)
    length = pl.col("sequence").str.len_chars()
    unknown_ratio = pl.col("sequence").str.count_matches("[^ACGU]") / length
    return (
        length.gt(0)
        & resolution_ok
        & pl.col("fraction_missing").le(Config3D.MAX_MISSING)
        & unknown_ratio.le(Config3D.MAX_UNKNOWN_RATIO)
    )


def monomer_filter() -> pl.Expr:
    """Select self-structured RNA monomers passing the quality filters."""
    return (
        quality_filter()
        & pl.col("polymer_coverage").le(Config3D.MAX_POLYMER_COVERAGE)
        & pl.col("self_structured")
    )
