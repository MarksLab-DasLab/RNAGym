"""Validated tables shared by fitness scoring and evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl


def parse_row_ids(specification: str) -> list[int]:
    """Expand reference rows such as ``0-8,11-32`` in ascending order."""
    rows = set()
    for selection in specification.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:-(\d+))?\s*", selection)
        if match is None:
            raise ValueError(f"Invalid --rows selection: {selection!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            raise ValueError(f"Descending --rows range: {selection!r}")
        rows.update(range(start, end + 1))
    return sorted(rows)


def read_assay(path: str | Path) -> pl.DataFrame:
    """Read an assay without dropping its wild-type or repeated measurements."""
    table = pl.read_csv(path)
    require_columns(table, ("mutant", "sequence", "DMS_score"), path)
    return table


def read_reference(path: str | Path, rows: str = "all") -> pl.DataFrame:
    """Read validated metadata for all assays or selected zero-based rows."""
    table = pl.read_csv(path)
    require_columns(table, ("DMS_ID", "RNA_TYPE", "RAW_CONSTRUCT_SEQ"), path)
    if table.is_empty():
        raise ValueError(f"{path} has no assays")
    for column in ("DMS_ID", "RNA_TYPE", "RAW_CONSTRUCT_SEQ"):
        if (
            table[column].is_null().any()
            or table[column].str.strip_chars().eq("").any()
        ):
            raise ValueError(f"{path} has missing {column} values")
    if table["DMS_ID"].is_duplicated().any():
        raise ValueError(f"{path} repeats DMS_ID values")
    if rows != "all":
        row_ids = parse_row_ids(rows)
        if row_ids[-1] >= table.height:
            raise ValueError(f"Rows {row_ids} fall outside the reference sheet")
        table = table[row_ids]
    return table


def require_columns(
    table: pl.DataFrame, columns: Sequence[str], path: str | Path
) -> None:
    """Reject an incomplete table schema before processing its rows."""
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def write_csv_atomically(table: pl.DataFrame, path: Path) -> None:
    """Replace a CSV only after the complete table has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    permissions = path.stat().st_mode & 0o777 if path.exists() else 0o644
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        table.write_csv(temporary)
        temporary.chmod(permissions)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
