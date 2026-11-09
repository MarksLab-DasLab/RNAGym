"""Stage the clinical benchmark files once for every scoring job.

Each source CSV is read once and checked against its published sha256. Every
row's alleles must match its sequences, and an SNV must change exactly one
base. Nuclear rows containing ``N`` are excluded. The mitochondrial reference
keeps an ``N`` at position 3107 as a numbering placeholder for a base that does
not exist, so it is deleted from both sequences and ``variant_index`` is shifted
when the placeholder lies upstream of the variant.

Output per file, in ``--out``: ``<name>.parquet`` (all metadata columns as
strings, plus ``row_id``, ``cohort``, ``group_id``, ``variant_index``,
``seq_ref``, ``seq_alt``), ``<name>.excluded.csv`` and ``<name>.summary.json``.
``row_id`` is the 0-based data row of the source CSV. ``group_id`` numbers
distinct reference sequences in order of first appearance, so rows sharing a
reference land in the same scoring shard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from rnagym.clinical.common import file_sha256
from rnagym.config import ConfigClinical

NUCLEAR = (
    "premrna_10101_snv",
    "mature_mrna_snv",
    "premrna_10101_indel",
    "mature_mrna_indel",
)
# The mitochondrial RNA view: mRNAs and rRNAs, and the tRNAs with their CCA tail
MITO = ("mt_rna", "mt_trna_cca")
BASES = frozenset("ACGT")


def sources(variant_dir: Path) -> dict[str, tuple[Path, str]]:
    """Map each staged name to its source CSV and published sha256."""
    nuclear = json.loads((variant_dir / "export_summary.json").read_text())
    mito = json.loads((variant_dir / "mito_export_summary.json").read_text())
    out = {
        name: (variant_dir / f"{name}.csv", nuclear[name]["sha256"]) for name in NUCLEAR
    }
    for name in MITO:
        out[name] = (variant_dir / f"{name}.csv", mito[name]["sha256"])
    return out


def check_row(
    seq_ref: str,
    seq_alt: str,
    vi: int,
    ref: str,
    alt: str,
    snv: bool,
    fixed_length: bool,
) -> None:
    """Raise unless the sequences differ by exactly the declared alleles at ``vi``.

    Upstream of the variant the sequences are identical. Downstream, the
    reference after its allele and the alternate after its allele agree over
    their shared length. In a fixed-length window (pre-mRNA) both sequences have
    the same length; in a whole-unit view (mRNA, mito RNA) the lengths differ by
    exactly the allele lengths.
    """
    if seq_ref[vi : vi + len(ref)] != ref:
        raise ValueError(f"reference allele {ref!r} not at index {vi}")
    if seq_alt[vi : vi + len(alt)] != alt:
        raise ValueError(f"alternate allele {alt!r} not at index {vi}")
    if seq_ref[:vi] != seq_alt[:vi]:
        raise ValueError("sequences differ upstream of the variant")
    if snv and (len(ref) != 1 or len(alt) != 1 or ref == alt):
        raise ValueError(f"not a substitution: {ref!r} > {alt!r}")
    tail_ref, tail_alt = seq_ref[vi + len(ref) :], seq_alt[vi + len(alt) :]
    shared = min(len(tail_ref), len(tail_alt))
    if tail_ref[:shared] != tail_alt[:shared]:
        raise ValueError("sequences differ downstream of the variant")
    if fixed_length:
        if len(seq_ref) != len(seq_alt):
            raise ValueError("fixed-length window with unequal sequence lengths")
    elif len(seq_alt) - len(seq_ref) != len(alt) - len(ref):
        raise ValueError("whole-unit lengths do not differ by the allele lengths")


def delete_placeholder(
    seq_ref: str, seq_alt: str, vi: int, ref: str, alt: str
) -> tuple[str, str, int, bool]:
    """Delete the single mitochondrial ``N`` placeholder from both sequences."""
    n_ref, n_alt = seq_ref.count("N"), seq_alt.count("N")
    if n_ref == 0 and n_alt == 0:
        return seq_ref, seq_alt, vi, False
    if n_ref != 1 or n_alt != 1:
        raise ValueError(f"expected one placeholder N, found {n_ref} and {n_alt}")
    i_ref, i_alt = seq_ref.index("N"), seq_alt.index("N")
    if vi <= i_ref < vi + len(ref) or vi <= i_alt < vi + len(alt):
        raise ValueError("the placeholder N lies inside the variant alleles")
    upstream = i_ref < vi
    if upstream != (i_alt < vi) or (upstream and i_ref != i_alt):
        raise ValueError("the placeholder N is not at the same place in both alleles")
    if not upstream and i_alt - i_ref != len(alt) - len(ref):
        raise ValueError("the placeholder N moved by other than the indel length")
    seq_ref = seq_ref[:i_ref] + seq_ref[i_ref + 1 :]
    seq_alt = seq_alt[:i_alt] + seq_alt[i_alt + 1 :]
    return seq_ref, seq_alt, vi - int(upstream), True


def stage(
    name: str, path: Path, expected_sha: str, out: Path, skip_sha: bool
) -> dict[str, Any]:
    """Stage one file and return its summary."""
    actual_sha = None if skip_sha else file_sha256(path)
    if actual_sha is not None and actual_sha != expected_sha:
        raise ValueError(f"{path} sha256 {actual_sha} != published {expected_sha}")
    mito = name.startswith("mt_")
    fixed_length = "10101" in name

    csv.field_size_limit(1 << 30)
    kept: dict[str, list[Any]] = {}
    excluded: list[tuple[int, str, str, str]] = []
    groups: dict[bytes, int] = {}
    deletions: Counter[tuple[str, str]] = Counter()
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first: dict[str, int] = {}
        for k, column in enumerate(header):
            first.setdefault(column, k)
        repeated = [k for k, column in enumerate(header) if first[column] != k]
        meta = [c for c in first if c not in ("seq_ref", "seq_alt", "variant_index")]
        columns = ("row_id", "cohort", "group_id", "variant_index", *meta)
        for column in (*columns, "seq_ref", "seq_alt"):
            kept[column] = []

        for row_id, row in enumerate(reader):
            for k in repeated:
                if row[k] != row[first[header[k]]]:
                    raise ValueError(
                        f"row {row_id}: repeated column {header[k]} disagrees"
                    )
            get = {column: row[k] for column, k in first.items()}
            label = get["binary_label"]
            if label not in ("0", "1"):
                raise ValueError(f"row {row_id}: binary_label {label!r}")
            seq_ref, seq_alt = get["seq_ref"], get["seq_alt"]
            vi = int(get["variant_index"])
            ref, alt = get["ref_allele_tx"], get["alt_allele_tx"]
            snv = get["variant_class"] == "snv"
            check_row(seq_ref, seq_alt, vi, ref, alt, snv, fixed_length)

            has_n = "N" in seq_ref or "N" in seq_alt
            if has_n != (get["has_n"] == "True"):
                raise ValueError(f"row {row_id}: has_n disagrees with the sequences")
            if has_n and not mito:
                excluded.append((row_id, get["variation_id"], label, "has_n"))
                continue
            if mito:
                seq_ref, seq_alt, vi, deleted = delete_placeholder(
                    seq_ref, seq_alt, vi, ref, alt
                )
                if deleted:
                    upstream = vi < int(get["variant_index"])
                    side = "upstream" if upstream else "downstream"
                    deletions[(get["variant_class"], side)] += 1
                check_row(seq_ref, seq_alt, vi, ref, alt, snv, fixed_length)
            if not set(seq_ref) <= BASES or not set(seq_alt) <= BASES:
                raise ValueError(f"row {row_id}: a base outside ACGT")

            key = hashlib.sha1(seq_ref.encode()).digest()
            group = groups.setdefault(key, len(groups))
            kept["row_id"].append(row_id)
            kept["cohort"].append("snv" if snv else "nonsnv")
            kept["group_id"].append(group)
            kept["variant_index"].append(vi)
            for column in meta:
                kept[column].append(get[column])
            kept["seq_ref"].append(seq_ref)
            kept["seq_alt"].append(seq_alt)

    frame = pl.DataFrame(kept, schema_overrides={c: pl.String for c in meta})
    frame.write_parquet(
        out / f"{name}.parquet", compression="zstd", row_group_size=2048
    )
    with (out / f"{name}.excluded.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "variation_id", "binary_label", "reason"))
        writer.writerows(excluded)

    counts = (
        frame.group_by("cohort", "binary_label")
        .len()
        .sort("cohort", "binary_label")
        .rows()
    )
    summary = {
        "source": str(path),
        "source_sha256": actual_sha,
        "sha256_checked": actual_sha is not None,
        "rows_in": len(frame) + len(excluded),
        "rows_kept": len(frame),
        "rows_excluded": dict(Counter(reason for *_, reason in excluded)),
        "kept_by_cohort_and_label": {f"{c}/{label}": n for c, label, n in counts},
        "reference_groups": len(groups),
        "placeholder_deletions": {
            f"{c}/{side}": n for (c, side), n in sorted(deletions.items())
        },
        "staged_sha256": file_sha256(out / f"{name}.parquet"),
    }
    (out / f"{name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=ConfigClinical.VARIANT_DIR)
    parser.add_argument("--out", type=Path, default=ConfigClinical.STAGED_DIR)
    parser.add_argument("--only", nargs="*", help="Stage only these names")
    parser.add_argument("--skip-sha", action="store_true", help="Tests only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Stage every released file, or the named ones."""
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (path, expected) in sources(args.input).items():
        if args.only and name not in args.only:
            continue
        summary = stage(name, path, expected, args.out, args.skip_sha)
        print(name, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
