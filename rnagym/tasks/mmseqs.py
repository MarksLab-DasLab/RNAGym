"""Generate standard nucleotide MMseqs2 MSAs for every existing benchmark MSA."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shlex import quote, split

import polars as pl

from rnagym.config import Config3D, ConfigFitness, ConfigMMseqs, ConfigRiboseek
from rnagym.sequences import fitness_assays, load_registry
from rnagym.tasks.prepare_mmseqs import run
from rnagym.tasks.riboseek import normalize_query, read_alignment

DESCRIPTION = "MMseqs2 nucleotide 30k x1"
COMPLEMENT = str.maketrans("ACGTURYSWKMBDHVNX-", "TGCAAYRSWMKVHDBNX-")
HIT_COLUMNS = "query,target,qstart,qend,qaln,taln,evalue,bits"


def _atomic_text(path: Path, text: str) -> None:
    """Replace a file after its entire content has been written."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _fingerprint(database: Path) -> dict[str, list[str | int]]:
    """Identify the exact sequence and header files used for a search."""
    result = {}
    for suffix in ("", ".index", ".dbtype", "_h", "_h.index", "_h.dbtype"):
        path = Path(f"{database}{suffix}").resolve(strict=True)
        stat = path.stat()
        result[suffix] = [str(path), stat.st_size, stat.st_mtime_ns]
    if database.with_suffix(".dbtype").read_bytes() != (1).to_bytes(4, sys.byteorder):
        raise ValueError(f"Expected a standard nucleotide database: {database}")
    return result


@contextmanager
def _lock(directory: Path) -> Iterator[None]:
    """Reject concurrent writers to one run or database search."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def inventory() -> pl.DataFrame:
    """Freeze the exact query sequences and paths of all existing MSAs."""
    registry = load_registry(ConfigMMseqs.SEQUENCE_FILE)
    rows = []
    for benchmark, source in (
        ("3d", Config3D.MSA_DIR),
        ("fitness", ConfigFitness.MSA_DIR),
    ):
        for path in sorted(source.glob("*.a3m")):
            rows.append((path.stem, benchmark, str(path.resolve())))
    if not rows:
        raise ValueError("No existing benchmark MSAs were found")
    queries = pl.DataFrame(
        rows, schema=["sequence_id", "benchmark", "source"], orient="row"
    ).join(registry.select("sequence_id", "sequence"), on="sequence_id", how="left")
    if queries["sequence"].null_count():
        raise ValueError("An existing MSA has no registered sequence")
    for row in queries.iter_rows(named=True):
        records = read_alignment(Path(row["source"]), max_records=1)
        if not records:
            raise ValueError(f"Empty source MSA: {row['source']}")
        query = "".join(c for c in records[0][1] if not c.islower()).replace("-", "")
        if records[0][0].split()[0] != f">{row['sequence_id']}" or normalize_query(
            query
        ) != normalize_query(row["sequence"]):
            raise ValueError(f"Incorrect query in {row['source']}")
    queries = queries.sort("sequence_id", "benchmark")
    ConfigMMseqs.WORK_DIR.mkdir(parents=True, exist_ok=True)
    if ConfigMMseqs.QUERY_FILE.exists():
        if not queries.equals(pl.read_parquet(ConfigMMseqs.QUERY_FILE)):
            raise ValueError("Existing MMseqs2 inventory differs from current MSAs")
    else:
        queries.write_parquet(ConfigMMseqs.QUERY_FILE)
    print(
        f"Inventoried {queries['sequence_id'].n_unique()} unique queries "
        f"for {queries.height} existing MSAs",
        flush=True,
    )
    return queries


def project_hit(sequence: str, start: int, end: int, query: str, target: str) -> str:
    """Orient one local nucleotide alignment and project it into query A3M space.

    Parameters
    ----------
    sequence : str
        Full query sequence in its original orientation.
    start, end : int
        One-based inclusive MMseqs2 query coordinates. Descending coordinates
        identify a reverse-complemented query alignment.
    query, target : str
        Equal-length gapped strings from nucleotide ``convertalis``.

    Returns
    -------
    str
        RNA A3M row with lowercase insertions and terminal gaps.
    """
    if len(query) != len(target) or not query:
        raise ValueError("Unequal or empty pairwise alignment strings")
    query, target = query.upper(), target.upper()
    # RNAcentral retains inosine as I, which becomes X during normalization
    if (set(query) | set(target)) - set("ACGTURYSWKMBDHVNXI-"):
        raise ValueError("Unexpected nucleotide in the pairwise alignment")
    if start > end:
        start, end = end, start
        query = query.translate(COMPLEMENT)[::-1]
        target = target.translate(COMPLEMENT)[::-1]
    if not 1 <= start <= end <= len(sequence):
        raise ValueError("Alignment coordinates exceed the query")
    if normalize_query(query.replace("-", "")) != normalize_query(
        sequence[start - 1 : end]
    ):
        raise ValueError("Aligned query does not match the registered sequence")
    aligned = []
    for q, t in zip(query, target):
        if q == t == "-":
            raise ValueError("Pairwise alignment contains a gap-gap column")
        base = "-" if t == "-" else normalize_query(t)
        aligned.append(base.lower() if q == "-" else base)
    return "-" * (start - 1) + "".join(aligned) + "-" * (len(sequence) - end)


def run_info() -> tuple[pl.DataFrame, Path, dict[str, object]]:
    """Return a stable run directory keyed by queries, parameters, and software."""
    queries = pl.read_parquet(ConfigMMseqs.QUERY_FILE)
    version = subprocess.check_output(split("mmseqs version"), text=True).strip()
    info: dict[str, object] = {
        "description": DESCRIPTION,
        "mmseqs_version": version,
        "query_sha256": hashlib.sha256(
            ConfigMMseqs.QUERY_FILE.read_bytes()
        ).hexdigest(),
        "rnacentral_release": ConfigRiboseek.RNACENTRAL_VERSION,
        "nt_release": ConfigRiboseek.NT_VERSION,
        "evalue": ConfigMMseqs.EVALUE,
        "kmer_length": ConfigMMseqs.KMER_LENGTH,
        "max_seqs": ConfigMMseqs.MAX_SEQS,
        "strands": ConfigMMseqs.STRANDS,
        "iterations": 1,
        "search_type": 3,
        "prefilter_mode": 0,
        "max_target_length": ConfigRiboseek.MAX_TARGET_LENGTH,
        "target_overlap": ConfigRiboseek.TARGET_OVERLAP,
    }
    digest = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]
    return queries, ConfigMMseqs.CACHE_DIR / digest, info


def search(index: int) -> None:
    """Search all frozen queries against one complete nucleotide database."""
    queries, work, info = run_info()
    database = ConfigMMseqs.DATABASES[index]
    name = ConfigMMseqs.DATABASE_NAMES[index]
    directory = work / name
    fingerprint = _fingerprint(database)
    provenance = {**info, "database": fingerprint}
    # JSON normalizes tuples to lists when checking a previous checkpoint
    serialized = json.dumps(provenance, sort_keys=True, indent=2) + "\n"
    with _lock(directory):
        marker = directory / "SUCCESS.json"
        hits = directory / "hits.tsv"
        if marker.is_file():
            if marker.read_text() != serialized or not hits.is_file():
                raise ValueError(f"Stale or incomplete search: {directory}")
            print(f"Search is complete: {name}", flush=True)
            return
        recorded = directory / "parameters.json"
        if recorded.exists() and recorded.read_text() != serialized:
            raise ValueError(f"Search inputs changed: {directory}")
        _atomic_text(recorded, serialized)
        sequences = (
            queries.select("sequence_id", "sequence").unique().sort("sequence_id")
        )
        fasta = directory / "queries.fa"
        _atomic_text(
            fasta,
            "".join(
                f">{key}\n{normalize_query(seq).replace('U', 'T').replace('X', 'N')}\n"
                for key, seq in sequences.iter_rows()
            ),
        )
        query_db = directory / "queries"
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        memory = os.environ.get("RNAGYM_MMSEQS_MEMORY", "100G")
        result = directory / "alignments"
        qdb, tdb, result_arg = (
            quote(str(path)) for path in (query_db, database, result)
        )
        run(f"mmseqs createdb {quote(str(fasta))} {qdb} --dbtype 2 --shuffle 0")
        run(
            f"mmseqs search {qdb} {tdb} {result_arg} {quote(str(directory / 'tmp'))} "
            f"--search-type 3 --strand {ConfigMMseqs.STRANDS[index]} "
            f"--num-iterations 1 --prefilter-mode 0 -k {ConfigMMseqs.KMER_LENGTH} "
            f"-a -e {ConfigMMseqs.EVALUE} --max-seqs {ConfigMMseqs.MAX_SEQS} "
            f"--max-seq-len {ConfigRiboseek.MAX_TARGET_LENGTH} "
            f"--sequence-overlap {ConfigRiboseek.TARGET_OVERLAP} "
            f"--db-load-mode 2 --split-memory-limit {quote(memory)} --threads {threads}"
        )
        temporary = hits.with_suffix(".tsv.tmp")
        # result2msa in the pinned release uses protein scoring internally
        # convertalis preserves nucleotide backtraces, insertions, and strand
        run(
            f"mmseqs convertalis {qdb} {tdb} {result_arg} {quote(str(temporary))} "
            f"--search-type 3 --format-output {HIT_COLUMNS} "
            f"--db-load-mode 2 --threads {threads}"
        )
        if _fingerprint(database) != fingerprint:
            raise ValueError(f"Database changed during search: {database}")
        temporary.replace(hits)
        _atomic_text(marker, serialized)
        print(f"Search is complete: {name}", flush=True)


def finish() -> None:
    """Merge database hits, export paired alignments, and record their depths."""
    queries, work, info = run_info()
    sequences = dict(queries.select("sequence_id", "sequence").unique().iter_rows())
    with _lock(work):
        hits = {key: [] for key in sequences}
        for index, name in enumerate(ConfigMMseqs.DATABASE_NAMES):
            directory = work / name
            marker = directory / "SUCCESS.json"
            expected = {**info, "database": _fingerprint(ConfigMMseqs.DATABASES[index])}
            if not marker.exists() or json.loads(marker.read_text()) != json.loads(
                json.dumps(expected)
            ):
                raise ValueError(f"Search is incomplete or stale: {name}")
            with (directory / "hits.tsv").open() as handle:
                for line in handle:
                    key, target, start, end, qaln, taln, evalue, bits = line.rstrip(
                        "\n"
                    ).split("\t")
                    evalue, bits = float(evalue), float(bits)
                    if (
                        not math.isfinite(evalue)
                        or not math.isfinite(bits)
                        or evalue < 0
                    ):
                        raise ValueError("Invalid alignment score")
                    aligned = project_hit(
                        sequences[key], int(start), int(end), qaln, taln
                    )
                    header = f">{name}|{target} evalue={evalue:g} bits={bits:g}"
                    hits[key].append((-bits, evalue, header, aligned))

        report = []
        for key, sequence in sorted(sequences.items()):
            query = normalize_query(sequence)
            records = [(f">{key} {DESCRIPTION}", query)]
            seen = {query}
            for _, _, header, aligned in sorted(hits.pop(key)):
                if aligned not in seen:
                    records.append((header, aligned))
                    seen.add(aligned)
            contents = "".join(f"{header}\n{seq}\n" for header, seq in records)
            for row in queries.filter(pl.col("sequence_id") == key).iter_rows(
                named=True
            ):
                source = Path(row["source"])
                output = source.parent.with_name("mmseqs")
                output.mkdir(parents=True, exist_ok=True)
                a3m = output / f"{key}.a3m"
                _atomic_text(a3m, contents)
                _atomic_text(output / f"{key}.fa", f">{key}\n{sequence}\n")
                afa = output / f"{key}.afa"
                temporary = afa.with_suffix(".afa.tmp")
                run(f"reformat.pl a3m fas {quote(str(a3m))} {quote(str(temporary))}")
                aligned_records = read_alignment(temporary)
                lengths = {len(seq) for _, seq in aligned_records}
                if len(aligned_records) != len(records) or len(lengths) != 1:
                    raise ValueError(f"Malformed aligned FASTA: {temporary}")
                afa_query = aligned_records[0][1].replace("-", "")
                if afa_query != query:
                    raise ValueError(f"Incorrect query in {temporary}")
                temporary.replace(afa)
                with source.open() as handle:
                    original_depth = sum(line.startswith(">") for line in handle)
                report.append(
                    {
                        "sequence_id": key,
                        "benchmark": row["benchmark"],
                        "length": len(sequence),
                        "riboseek_depth": original_depth,
                        "mmseqs_depth": len(records),
                        "query_only": len(records) == 1,
                        "a3m": str(a3m),
                    }
                )

        aliases = fitness_assays().join(
            queries.filter(pl.col("benchmark") == "fitness"), on="sequence"
        )
        for row in aliases.iter_rows(named=True):
            directory = Path(row["source"]).parent.with_name("mmseqs") / "by_assay"
            directory.mkdir(exist_ok=True)
            for suffix in (".a3m", ".afa", ".fa"):
                link = directory / f"{row['DMS_ID']}{suffix}"
                target = Path("..") / f"{row['sequence_id']}{suffix}"
                if link.is_symlink() and link.readlink() == target:
                    continue
                link.symlink_to(target)
        pl.DataFrame(report).sort("benchmark", "sequence_id").write_csv(
            ConfigMMseqs.WORK_DIR / "summary.csv"
        )
        info["searches"] = [
            json.loads((work / name / "SUCCESS.json").read_text())
            for name in ConfigMMseqs.DATABASE_NAMES
        ]
        info["unique_queries"] = len(sequences)
        info["paired_msas"] = len(report)
        _atomic_text(
            ConfigMMseqs.WORK_DIR / "SUCCESS.json", json.dumps(info, indent=2) + "\n"
        )
        print(
            f"Wrote {len(report)} paired MSAs for {len(sequences)} unique queries",
            flush=True,
        )


def main() -> None:
    """Inventory, search, or finish the conventional nucleotide MSA baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inventory", "search", "finish"))
    args = parser.parse_args()
    if args.stage == "inventory":
        inventory()
    elif args.stage == "search":
        search(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    else:
        finish()


if __name__ == "__main__":
    main()
