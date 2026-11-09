"""Shard staged rows, crop allele pairs, and write outputs with provenance."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import uuid
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from rnagym.clinical.windows import crop_bounds, variant_span
from rnagym.config import ConfigClinical

COLUMNS = (
    "row_id",
    "cohort",
    "group_id",
    "variant_index",
    "ref_allele_tx",
    "alt_allele_tx",
    "seq_ref",
    "seq_alt",
)


def shard_of_groups(
    group_ids: npt.NDArray[np.int64], num_shards: int
) -> npt.NDArray[np.int64]:
    """Assign each row a shard, never splitting rows that share a reference.

    Groups are ordered by id and cut into contiguous blocks of about equal row
    counts, so the assignment depends only on the staged file and ``num_shards``.
    """
    groups, counts = np.unique(group_ids, return_counts=True)
    before = np.concatenate(([0], np.cumsum(counts)[:-1]))
    shard_of_group = (before * num_shards) // len(group_ids)
    return shard_of_group[np.searchsorted(groups, group_ids)]


def requested_ids(
    row_ids: Path, staged_ids: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Read a row id list and fail unless every id is a staged row."""
    tokens = row_ids.read_text().split()
    wanted = np.array(sorted({int(x) for x in tokens}), dtype=np.int64)
    if len(wanted) != len(tokens):
        raise ValueError(f"{row_ids} lists a row id twice")
    if len(wanted) == 0:
        raise ValueError(f"{row_ids} is empty")
    missing = wanted[~np.isin(wanted, staged_ids)]
    if len(missing):
        raise ValueError(
            f"{row_ids} names {len(missing)} rows that are not staged, "
            f"e.g. {missing[:5]}"
        )
    return wanted


def load_rows(
    staged: Path, name: str, shard: int, num_shards: int, row_ids: Path | None
) -> pl.DataFrame:
    """Read one shard of a staged file, optionally restricted to listed rows."""
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} outside 0..{num_shards - 1}")
    frame = pl.read_parquet(staged / f"{name}.parquet", columns=list(COLUMNS))
    wanted = None
    if row_ids is not None:
        wanted = requested_ids(row_ids, frame["row_id"].to_numpy())
    keep = shard_of_groups(frame["group_id"].to_numpy(), num_shards) == shard
    frame = frame.filter(pl.Series(keep))
    if wanted is not None:
        frame = frame.filter(pl.col("row_id").is_in(wanted))
    return frame


def expected_row_ids(
    staged: Path, name: str, row_ids: Path | None
) -> npt.NDArray[np.int64]:
    """Return the row ids every shard together must cover."""
    ids = np.sort(
        pl.read_parquet(staged / f"{name}.parquet", columns=["row_id"])[
            "row_id"
        ].to_numpy()
    )
    if row_ids is not None:
        return requested_ids(row_ids, ids)
    return ids


@dataclass
class Crop:
    """The reference and alternate windows of one row for one model."""

    start: int
    ref: str
    alt: str
    site: int  # variant_index inside the windows
    contained: bool


def _contains(seq_len: int, start: int, end: int, vi: int, allele_len: int) -> bool:
    """Whether an allele, or the flanks of an empty one, lie inside a window."""
    if allele_len:
        return start <= vi and vi + allele_len <= end
    left_ok = vi == 0 or start <= vi - 1
    right_ok = vi == seq_len or vi < end
    return left_ok and right_ok


def crop_pair(
    seq_ref: str, seq_alt: str, vi: int, ref: str, alt: str, cap: int
) -> Crop:
    """Crop both alleles to ``cap`` at a shared start centred on the variant."""
    lo, hi = variant_span(vi, ref, len(seq_ref))
    if hi - lo + 1 > cap:
        # The reference allele alone is longer than the window: crop at its
        # start so the row is still described, and mark it as not contained
        hi = lo
    start, end = crop_bounds(len(seq_ref), lo, hi, cap)
    alt_end = min(len(seq_alt), start + cap)
    contained = _contains(len(seq_ref), start, end, vi, len(ref)) and _contains(
        len(seq_alt), start, alt_end, vi, len(alt)
    )
    return Crop(
        start, seq_ref[start:end], seq_alt[start:alt_end], vi - start, contained
    )


def package_sources() -> list[Path]:
    """Return every Python file of the rnagym package that scoring can import.

    Hidden directories are skipped: the benchmarks' ``.pixi`` environments live
    inside the package tree and hold libraries, not rnagym sources.
    """
    root = Path(__file__).resolve().parents[1]
    found: list[Path] = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [d for d in subdirectories if not d.startswith(".")]
        found += [Path(directory) / name for name in files if name.endswith(".py")]
    return sorted(found)


def source_sha256(paths: list[Path]) -> str:
    """Hash the scoring sources by relative path and content."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.resolve().relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class SourceGuard:
    """Fingerprint the sources before a model is loaded; refuse to publish if they change."""

    def __init__(self) -> None:
        """Hash the sources now."""
        self.sha256 = source_sha256(package_sources())

    def check(self) -> None:
        """Raise if any source changed since the guard was made."""
        if source_sha256(package_sources()) != self.sha256:
            raise RuntimeError(
                "scoring sources changed while this job ran; not publishing"
            )


def stack_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    """Return the installed version of each package, or None."""
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def file_sha256(path: Path) -> str:
    """Return the sha256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_key(path: Path) -> str:
    """Key a file by everything that changes when it is replaced or edited."""
    stat = path.stat()
    fields = (path, stat.st_dev, stat.st_ino, stat.st_size)
    times = (stat.st_mtime_ns, stat.st_ctime_ns)
    return "|".join(str(field) for field in (*fields, *times))


def checkpoint_digests(
    files: list[Path], cache: Path = ConfigClinical.CHECKPOINT_HASH_FILE
) -> dict[Path, str]:
    """Return each file's sha256, hashing only files the cache does not know.

    The cache maps a file's resolved path, device, inode, size, mtime and ctime
    to its digest, so a moved, replaced or edited file is hashed again. The lock
    is held while hashing, so concurrent jobs hash a new checkpoint once.
    Deleting the cache forces every file to be hashed again.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        known: dict[str, str] = json.loads(cache.read_text()) if cache.exists() else {}
        out = {}
        for path in files:
            key = _identity_key(path.resolve())
            if key not in known:
                known[key] = file_sha256(path)
            out[path] = known[key]
        temporary = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(known, indent=2, sort_keys=True) + "\n")
        temporary.replace(cache)
    return out


def file_identity(path: Path) -> str:
    """Return the identity of the file a path resolves to."""
    return _identity_key(path.resolve())


def _read_marks(cache: Path) -> set[str]:
    marks: list[str] = json.loads(cache.read_text()) if cache.exists() else []
    return set(marks)


def known_good(path: Path, check: str, cache: Path) -> bool:
    """Whether the file at ``path``, by identity, passed ``check`` before."""
    return path.is_file() and f"{check}|{file_identity(path)}" in _read_marks(cache)


def mark_good(identity: str, check: str, cache: Path) -> None:
    """Record that the file with this identity passed ``check``."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marks = _read_marks(cache) | {f"{check}|{identity}"}
        temporary = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}")
        temporary.write_text(json.dumps(sorted(marks), indent=2) + "\n")
        temporary.replace(cache)


def checkpoint_identity(
    path_or_id: str | Path,
    revision: str | None,
    cache: Path = ConfigClinical.CHECKPOINT_HASH_FILE,
) -> dict[str, Any]:
    """Identify a checkpoint: a pinned Hub revision, or a local file or directory."""
    path = Path(path_or_id)
    if not path.exists():
        if revision is None:
            raise ValueError(f"unpinned Hub checkpoint {path_or_id!r}")
        return {"hub_id": str(path_or_id), "revision": revision}
    files = sorted(
        p for p in ([path] if path.is_file() else path.rglob("*")) if p.is_file()
    )
    digests = checkpoint_digests(files, cache)
    record = []
    for file in files:
        stat = file.resolve().stat()
        record.append(
            {
                "path": str(file),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digests[file],
            }
        )
    return {"local": record}


def spec_hash(spec: dict[str, Any]) -> str:
    """Return a short hash of a scoring spec."""
    text = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def write_shard(
    out_dir: Path,
    shard: int,
    num_shards: int,
    frame: pl.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    """Publish a shard and its manifest under an exclusive claim; never overwrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{shard:04d}_of_{num_shards:04d}"
    try:
        fd = os.open(
            out_dir / f"{stem}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
    except FileExistsError:
        raise FileExistsError(
            f"{out_dir / stem} is already claimed; use a fresh output directory"
        ) from None
    with os.fdopen(fd, "w") as claim:
        job = os.environ.get("SLURM_JOB_ID", "")
        claim.write(f"{socket.gethostname()} {os.getpid()} {job}\n")
    existing = [
        p for p in (out_dir / f"{stem}.parquet", out_dir / f"{stem}.json") if p.exists()
    ]
    if existing:
        raise FileExistsError(
            f"{existing[0]} exists without a claim; refusing to overwrite it"
        )
    unique = uuid.uuid4().hex
    tmp = out_dir / f".{stem}.{unique}.parquet"
    frame.write_parquet(tmp)
    manifest = {
        **manifest,
        "output_sha256": hashlib.sha256(tmp.read_bytes()).hexdigest(),
    }
    tmp_json = out_dir / f".{stem}.{unique}.json"
    tmp_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    tmp_json.replace(out_dir / f"{stem}.json")
    target = out_dir / f"{stem}.parquet"
    tmp.replace(target)
    return target


def staged_sha256(staged: Path, name: str) -> str:
    """Return the staged file's digest, recomputed and checked against its summary."""
    summary = json.loads((staged / f"{name}.summary.json").read_text())
    recorded: str = summary["staged_sha256"]
    if file_sha256(staged / f"{name}.parquet") != recorded:
        raise ValueError(f"staged {name}.parquet does not match its summary")
    return recorded
