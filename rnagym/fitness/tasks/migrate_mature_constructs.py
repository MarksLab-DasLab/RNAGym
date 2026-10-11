"""Rewrite assay tables for constructs corrected to the mature molecule."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rnagym.config import ConfigFitness

MUTATION = re.compile(r"^([ACGUT])(\d+)([ACGUT])$")


@dataclass(frozen=True)
class Recipe:
    """One construct's mature-molecule recipe.

    Attributes
    ----------
    segments : tuple[tuple[int, int], ...]
        Retained spans as one-based inclusive bounds, in mature order.
    append : str
        Sequence added to the three-prime end after the segments.
    reason : str
        Short description printed when the assay is rewritten.
    note : str
        Sentence appended to the reference sheet's ``Notes`` column.
    """

    segments: tuple[tuple[int, int], ...]
    append: str
    reason: str
    note: str


RECIPES: dict[str, Recipe] = {
    "Guy_2014_tRNA": Recipe(
        segments=((1, 39), (54, 89)),
        append="CCA",
        reason="intron 40-53 spliced; 3' flank 90-105 removed; mature CCA added",
        note="Mature tRNA: intron 40-53 spliced, 3' flank 90-105 removed, CCA added.",
    ),
    "Domingo_2018_tRNA": Recipe(
        segments=((1, 72),),
        append="CCA",
        reason="mature CCA added",
        note="Mature tRNA: CCA added.",
    ),
    "Li_2016_tRNA": Recipe(
        segments=((1, 72),),
        append="CCA",
        reason="mature CCA added",
        note="Mature tRNA: CCA added.",
    ),
}

# Kobori 2018 stored 26 T and 4 U in one sequence, carrying RNA letters only where the
# scored positions forced them. Its assay table is already RNA, so the stored construct
# reconstructs none of its rows until T becomes U. No assay table changes.
TRANSCRIBE: dict[str, str] = {
    "Kobori_2018_ribozyme": "Construct transcribed to RNA (T -> U); assay data was already RNA.",
}

Row = dict[str, str]
Mutation = tuple[str, int, str]


def build_mapping(
    segments: Sequence[tuple[int, int]], construct_length: int
) -> dict[int, int]:
    """Map retained one-based positions to their mature positions."""
    mapping: dict[int, int] = {}
    new_position = 0
    for start, end in segments:
        if start < 1 or end > construct_length or start > end:
            raise ValueError(f"segment {start}-{end} outside 1-{construct_length}")
        for old in range(start, end + 1):
            new_position += 1
            mapping[old] = new_position
    return mapping


def mature_sequence(
    construct: str, segments: Sequence[tuple[int, int]], append: str
) -> str:
    """Join the retained segments and add the mature three-prime end."""
    return "".join(construct[start - 1 : end] for start, end in segments) + append


def parse_mutations(label: str) -> list[Mutation]:
    """Parse one comma-separated substitution label into RNA mutations."""
    # An empty label is a wild-type control row, not an error: Andreasson 2020 carries one
    if not label.strip():
        return []
    parsed: list[Mutation] = []
    for token in label.split(","):
        match = MUTATION.match(token.strip())
        if not match:
            raise ValueError(f"unparsable mutation {token.strip()!r}")
        wild, position, mutant = match.groups()
        parsed.append((wild.replace("T", "U"), int(position), mutant.replace("T", "U")))
    positions = [position for _, position, _ in parsed]
    repeated = sorted({p for p in positions if positions.count(p) > 1})
    if repeated:
        # Two tokens for one position contradict each other, and applying them in
        # order would silently let the last one win
        raise ValueError(f"label {label!r} names position(s) {repeated} more than once")
    return parsed


def migrate_assay(
    rows: Iterable[Row], construct: str, recipe: Recipe
) -> tuple[list[Row], str]:
    """Rewrite one assay's rows onto its mature construct."""
    mapping = build_mapping(recipe.segments, len(construct))
    new_construct = mature_sequence(construct, recipe.segments, recipe.append)

    for old, new in mapping.items():
        if construct[old - 1] != new_construct[new - 1]:
            raise ValueError(f"mapping corrupts base at old {old} -> new {new}")

    new_rows: list[Row] = []
    for index, row in enumerate(rows, start=2):
        mutations = parse_mutations(row["mutant"])

        applied = list(construct)
        for wild, position, mutant in mutations:
            if not 1 <= position <= len(construct):
                raise ValueError(
                    f"line {index}: {wild}{position}{mutant} outside "
                    f"construct 1-{len(construct)}"
                )
            if construct[position - 1] != wild:
                raise ValueError(
                    f"line {index}: {wild}{position}{mutant} disagrees with "
                    f"construct base {construct[position - 1]}"
                )
            applied[position - 1] = mutant
        if row.get("sequence") and "".join(applied) != row["sequence"]:
            raise ValueError(
                f"line {index}: stored sequence does not match construct plus mutations"
            )

        dropped = [p for _, p, _ in mutations if p not in mapping]
        if dropped:
            raise ValueError(
                f"line {index}: positions {dropped} fall in a removed region"
            )

        mature = list(new_construct)
        for _, position, mutant in mutations:
            mature[mapping[position] - 1] = mutant

        new_row = dict(row)
        new_row["mutant"] = ",".join(
            f"{wild}{mapping[position]}{mutant}" for wild, position, mutant in mutations
        )
        new_row["sequence"] = "".join(mature)
        new_rows.append(new_row)

    return new_rows, new_construct


@dataclass(frozen=True)
class Table:
    """One CSV's contents and the byte conventions it was stored with.

    The reference sheet carries a UTF-8 BOM and CRLF endings while the assay
    tables carry neither, so each file is rewritten the way it was read and an
    untouched row stays byte-identical.
    """

    columns: list[str]
    rows: list[Row]
    bom: bool
    newline: str


def read_csv(path: Path) -> Table:
    """Read one CSV, rejecting a blank header name or a ragged row."""
    head = path.read_bytes()[:4096]
    bom = head.startswith(b"\xef\xbb\xbf")
    newline = "\r\n" if b"\r\n" in head else "\n"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(not name for name in reader.fieldnames):
            raise ValueError(f"{path}: blank column name in header")
        columns = list(reader.fieldnames)
        rows: list[Row] = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"{path}: a row has more or fewer fields than the header"
                )
            rows.append({key: value for key, value in row.items() if key is not None})
    return Table(columns=columns, rows=rows, bom=bom, newline=newline)


def write_csv(path: Path, table: Table) -> None:
    """Write one CSV atomically, in the byte conventions it was read with."""
    temporary = path.with_suffix(".csv.partial")
    encoding = "utf-8-sig" if table.bom else "utf-8"
    with temporary.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=table.columns, lineterminator=table.newline
        )
        writer.writeheader()
        writer.writerows(table.rows)
    temporary.replace(path)


def read_reference(path: Path) -> Table:
    """Read the reference sheet, rejecting a duplicate assay identifier."""
    table = read_csv(path)
    seen: set[str] = set()
    for row in table.rows:
        if row["DMS_ID"] in seen:
            raise ValueError(f"{path}: duplicate DMS_ID {row['DMS_ID']}")
        seen.add(row["DMS_ID"])
    return table


def annotate(row: Row, note: str) -> None:
    """Append one sentence to a reference row's notes."""
    existing = (row.get("Notes") or "").strip()
    row["Notes"] = f"{existing} {note}" if existing else note


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite every corrected assay, or write nothing and report why."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assays-dir", type=Path, default=ConfigFitness.ASSAY_DIR)
    parser.add_argument("--reference", type=Path, default=ConfigFitness.REFERENCE_FILE)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--out-reference", type=Path, help="write the corrected reference sheet here"
    )
    args = parser.parse_args(argv)

    if args.assays_dir.resolve() == args.out_dir.resolve():
        print(
            "refusing to write into the source directory: that would overwrite the "
            "originals in place",
            file=sys.stderr,
        )
        return 1

    try:
        reference_table = read_reference(args.reference)
    except ValueError as error:
        print(f"FAILED {error}", file=sys.stderr)
        return 1
    reference = {row["DMS_ID"]: row for row in reference_table.rows}

    # Migrate in memory first. A partial batch would leave the published data a mixture
    # of new, stale and missing tables, so nothing is written until all of it succeeds.
    staged: list[tuple[str, Table, int, str, Recipe]] = []
    failures: list[str] = []

    for assay, recipe in sorted(RECIPES.items()):
        source = args.assays_dir / f"{assay}.csv"
        if assay not in reference:
            failures.append(f"{assay}: absent from reference sheet")
            continue
        if not source.is_file():
            failures.append(f"{assay}: {source} not found")
            continue

        construct = reference[assay]["RAW_CONSTRUCT_SEQ"]
        try:
            table = read_csv(source)
            new_rows, new_construct = migrate_assay(table.rows, construct, recipe)
        except ValueError as error:
            failures.append(f"{assay}: {error}")
            continue
        staged.append(
            (
                assay,
                replace(table, rows=new_rows),
                len(construct),
                new_construct,
                recipe,
            )
        )

    if failures:
        for failure in failures:
            print(f"FAILED {failure}", file=sys.stderr)
        print("nothing was written", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for assay, table, old_length, new_construct, recipe in staged:
        write_csv(args.out_dir / f"{assay}.csv", table)
        print(
            f"{assay}: {old_length} -> {len(new_construct)} nt, "
            f"{len(table.rows)} rows ({recipe.reason})"
        )

    if args.out_reference:
        constructs = {assay: sequence for assay, _, _, sequence, _ in staged}
        for row in reference_table.rows:
            assay = row["DMS_ID"]
            if assay in constructs:
                row["RAW_CONSTRUCT_SEQ"] = constructs[assay]
                annotate(row, RECIPES[assay].note)
            elif assay in TRANSCRIBE and "T" in row["RAW_CONSTRUCT_SEQ"]:
                row["RAW_CONSTRUCT_SEQ"] = row["RAW_CONSTRUCT_SEQ"].replace("T", "U")
                annotate(row, TRANSCRIBE[assay])
                print(f"{assay}: construct transcribed T -> U (reference sheet only)")
        write_csv(args.out_reference, reference_table)
        print(f"reference sheet: {args.out_reference} ({len(constructs)} rows updated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
