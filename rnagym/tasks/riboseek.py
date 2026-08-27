"""Generate shared Riboseek MSAs for 3D and fitness sequences."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from shlex import split

import pandas as pd

from rnagym.config import Config3D, ConfigFitness, ConfigRiboseek
from rnagym.sequences import fitness_assays, fitness_sequences, load_registry

MSA_DESCRIPTION = "Riboseek CM 30k x3"


def normalize_query(sequence: str) -> str:
    """Represent every noncanonical RNA residue as X."""
    return "".join(
        base if base in "ACGU" else "X" for base in sequence.replace("T", "U")
    )


def read_alignment(path: Path, max_records: int | None = None) -> list[tuple[str, str]]:
    """Read aligned FASTA records while joining wrapped sequence lines."""
    records: list[list[str]] = []
    with path.open() as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if max_records is not None and len(records) == max_records:
                    break
                records.append([line, ""])
            elif records:
                records[-1][1] += line
    return [(header, sequence) for header, sequence in records]


def valid_alignment(path: Path, sequence_id: str, sequence: str) -> bool:
    """Check one alignment's header and normalized query sequence."""
    if not path.is_file():
        return False
    try:
        records = read_alignment(path, max_records=1)
        query = "".join(base for base in records[0][1] if not base.islower())
        return records[0][0] == f">{sequence_id} {MSA_DESCRIPTION}" and query.replace(
            "-", ""
        ).replace("T", "U") == normalize_query(sequence)
    except (IndexError, OSError):
        return False


def is_complete(path: Path, sequence_id: str, sequence: str) -> bool:
    """Check the A3M, aligned FASTA, and query FASTA for one sequence."""
    fasta = path.with_suffix(".fa")
    return (
        valid_alignment(path, sequence_id, sequence)
        and valid_alignment(path.with_suffix(".afa"), sequence_id, sequence)
        and fasta.is_file()
        and fasta.read_text() == f">{sequence_id}\n{sequence}\n"
    )


def _link_fitness_alignments(sequence_ids: set[str]) -> None:
    """Link assay keys to their shared sequence alignments."""
    alias_dir = ConfigFitness.MSA_DIR / "by_assay"
    alias_dir.mkdir(parents=True, exist_ok=True)
    assays = fitness_assays().join(
        load_registry(ConfigRiboseek.SEQUENCE_FILE), on="sequence"
    )
    for assay, sequence_id in assays.select("DMS_ID", "sequence_id").iter_rows():
        if sequence_id not in sequence_ids:
            continue
        for suffix in (".a3m", ".afa", ".fa"):
            link = alias_dir / f"{assay}{suffix}"
            link.unlink(missing_ok=True)
            link.symlink_to(Path("..") / f"{sequence_id}{suffix}")


def remove_db(path: Path) -> None:
    """Remove a Riboseek database if it exists."""
    subprocess.run(
        split(f"riboseek rmdb {path}"),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_search(command: str, output: Path) -> None:
    """Run and checkpoint one expensive database search."""
    marker = output.with_suffix(".done")
    if marker.is_file():
        return
    remove_db(output)
    subprocess.run(split(command), check=True)
    marker.touch()


def run_cmsearch(cm: Path, output: Path, worker: int | None, workers: int) -> None:
    """Search one covariance model batch or merge all completed batches."""
    marker = output.with_suffix(".done")
    if marker.is_file():
        return
    # A different worker count defines different CM subsets
    part_dir = cm.parent / f"cmsearch_{workers}"
    part_dir.mkdir(exist_ok=True)
    keys = [
        line.split("\t", 1)[0]
        for line in cm.with_suffix(".index").read_text().splitlines()
    ]
    alignments = [part_dir / f"worker_{index}_alignment" for index in range(workers)]
    if worker is None:
        missing = [
            path for path in alignments if not path.with_suffix(".done").is_file()
        ]
        if missing:
            raise RuntimeError(f"{len(missing)} of {workers} CM workers failed")
        remove_db(output)
        subprocess.run(
            split(f"riboseek mergedbs {cm} {output} {' '.join(map(str, alignments))}"),
            check=True,
        )
        marker.touch()
        return

    key_file = part_dir / f"worker_{worker}.txt"
    query_cm = part_dir / f"worker_{worker}_cm"
    result = part_dir / f"worker_{worker}_result"
    alignment = alignments[worker]
    if alignment.with_suffix(".done").is_file():
        print(f"CM worker {worker + 1}/{workers} is complete")
        return
    key_file.write_text("\n".join(keys[worker::workers]) + "\n")
    for source, subset in (
        (cm, query_cm),
        (Path(f"{cm}_result_merged"), result),
    ):
        remove_db(subset)
        subprocess.run(
            split(
                f"riboseek createsubdb {key_file} {source} {subset} --subdb-mode 1 -v 1"
            ),
            check=True,
        )
    threads = os.environ.get("SLURM_CPUS_PER_TASK", "16")
    run_search(
        f"riboseek cmsearch {query_cm} {cm}_target_merged "
        f"{result} {alignment} --cm-region 3.0 -e inf --threads {threads}",
        alignment,
    )
    print(f"CM worker {worker + 1}/{workers} finished")


def load_queries(task_id: int, task_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one shard of unique sequences and their output directories."""
    targets = (
        pd.read_parquet(Config3D.TARGET_FILE)[["sequence_id", "sequence"]]
        .drop_duplicates()
        .assign(msa_dir=Config3D.MSA_DIR)
    )
    fitness_source = fitness_sequences()
    fitness = fitness_source.join(
        load_registry(ConfigRiboseek.SEQUENCE_FILE), on="sequence"
    )
    if fitness.height != fitness_source.height:
        raise RuntimeError("Run 'pixi run split' to register the fitness sequences")
    fitness = pd.DataFrame(
        fitness.select("sequence_id", "sequence").to_dict(as_series=False)
    ).assign(msa_dir=ConfigFitness.MSA_DIR)
    queries = pd.concat([targets, fitness]).drop_duplicates()
    sequences = queries[["sequence_id", "sequence"]].drop_duplicates()
    if sequences["sequence_id"].duplicated().any():
        raise RuntimeError("Each sequence ID must identify exactly one sequence")
    sequences = sequences.sort_values("sequence_id").iloc[task_id::task_count]
    return sequences, queries[queries["sequence_id"].isin(sequences["sequence_id"])]


def main() -> None:
    """Search RNAcentral and nt for one query shard, then write merged MSAs."""
    stage = sys.argv[1]
    task_id, task_count = map(int, sys.argv[2:4])
    threads = os.environ.get("SLURM_CPUS_PER_TASK", "32")
    sequences, queries = load_queries(task_id, task_count)

    for directory in queries["msa_dir"].unique():
        directory.mkdir(parents=True, exist_ok=True)
    missing_ids = {
        row.sequence_id
        for row in queries.itertuples(index=False)
        if not is_complete(
            row.msa_dir / f"{row.sequence_id}.a3m", row.sequence_id, row.sequence
        )
    }
    missing = sequences[sequences["sequence_id"].isin(missing_ids)]
    if stage == "check":
        raise SystemExit(not missing.empty)
    if missing.empty:
        _link_fitness_alignments(set(sequences["sequence_id"]))
        print(f"Riboseek shard {task_id + 1}/{task_count} is complete")
        return
    # Search only sequences without complete outputs
    sequences = missing

    batch = hashlib.sha256(
        "\n".join(
            [
                MSA_DESCRIPTION,
                ConfigRiboseek.RNACENTRAL_VERSION,
                ConfigRiboseek.NT_VERSION,
                *sequences["sequence_id"],
            ]
        ).encode()
    ).hexdigest()[:12]
    work_dir = ConfigRiboseek.CACHE_DIR / batch
    if stage == "ready":
        raise SystemExit(not (work_dir / "cm.done").is_file())
    if stage not in {"search", "cmsearch", "finish"}:
        raise ValueError(f"Unknown Riboseek stage: {stage}")

    work_dir.mkdir(parents=True, exist_ok=True)
    cm = work_dir / "cm"
    if stage == "cmsearch":
        if not cm.with_suffix(".done").is_file():
            raise RuntimeError(f"Riboseek search is incomplete for shard {task_id}")
        worker, workers = map(int, sys.argv[4:6])
        run_cmsearch(cm, work_dir / "cm_alignments", worker, workers)
        return

    databases = (ConfigRiboseek.DATABASE_RNACENTRAL, *ConfigRiboseek.DATABASE_NT_PARTS)
    if not ConfigRiboseek.DATABASE_NT_PARTS or any(
        not database.with_suffix(".dbtype").is_file() for database in databases
    ):
        raise FileNotFoundError("Run 'pixi run riboseek-db' first")

    fasta = work_dir / "queries.fasta"
    fasta.write_text(
        "".join(
            f">{sequence_id}\n{normalize_query(sequence)}\n"
            for sequence_id, sequence in sequences.itertuples(index=False)
        )
    )
    query_db = work_dir / "queries"

    if not missing.empty:
        try:
            run_search(
                f"riboseek createdb {fasta} {query_db} --threads {threads}", query_db
            )

            rnacentral = ConfigRiboseek.DATABASE_RNACENTRAL
            rnacentral_alignments = work_dir / "rnacentral_alignments"
            search_tmp = Path(os.environ.get("TMPDIR", "/tmp")) / (
                f"rnagym-riboseek-{batch}-rnacentral"
            )
            run_search(
                f"riboseek search {query_db} {rnacentral} "
                f"{rnacentral_alignments} {search_tmp} -a --gpu 1 "
                "--db-load-mode 2 --max-seqs 30000 --num-iterations 3 "
                f"--prefilter-mode 1 -e 0.1 --strand 1 --threads {threads}",
                rnacentral_alignments,
            )
            shutil.rmtree(search_tmp, ignore_errors=True)

            profile = work_dir / "profile"
            run_search(
                f"riboseek result2profile {query_db} {rnacentral} "
                f"{rnacentral_alignments} {profile} --e-profile 0.1 "
                f"--threads {threads}",
                profile,
            )

            databases = [rnacentral]
            alignment_dbs = [rnacentral_alignments]
            for index, nt in enumerate(ConfigRiboseek.DATABASE_NT_PARTS):
                alignment = work_dir / f"nt_{index}_alignments"
                search_tmp = Path(os.environ.get("TMPDIR", "/tmp")) / (
                    f"rnagym-riboseek-{batch}-nt-{index}"
                )
                run_search(
                    f"riboseek search {profile} {nt} {alignment} {search_tmp} "
                    "-a --gpu 1 --db-load-mode 2 --max-seqs 30000 "
                    "--num-iterations 1 --prefilter-mode 1 "
                    f"-e 0.1 --strand 2 --threads {threads}",
                    alignment,
                )
                shutil.rmtree(search_tmp, ignore_errors=True)
                databases.append(nt)
                alignment_dbs.append(alignment)

            # Build a covariance model from all hits and use it to realign them
            run_search(
                f"riboseek cmbuild {query_db} {','.join(map(str, databases))} "
                f"{','.join(map(str, alignment_dbs))} {cm} "
                f"--cmlite-msa-eval 1e-3 --threads {threads}",
                cm,
            )
            if stage == "search":
                print(f"Riboseek search finished for shard {task_id + 1}/{task_count}")
                return
            cm_alignments = work_dir / "cm_alignments"
            run_cmsearch(cm, cm_alignments, None, int(sys.argv[4]))
            msa_db = work_dir / "msas"
            unpacked = work_dir / "unpacked"
            commands = [
                f"riboseek result2msa {query_db} {cm}_target_merged "
                f"{cm_alignments} {msa_db} --msa-format-mode 6",
                f"riboseek unpackdb {msa_db} {unpacked} "
                "--unpack-suffix .a3m --unpack-name-mode 0",
            ]
            for command in commands:
                subprocess.run(split(f"{command} --threads {threads}"), check=True)

            for line in query_db.with_suffix(".lookup").read_text().splitlines():
                key, sequence_id, _ = line.split("\t")
                records = read_alignment(unpacked / f"{key}.a3m")
                records[0] = (f">{sequence_id} {MSA_DESCRIPTION}", records[0][1])
                contents = "".join(
                    f"{header}\n{aligned}\n" for header, aligned in records
                )
                for directory in queries.loc[
                    queries["sequence_id"] == sequence_id, "msa_dir"
                ]:
                    output = directory / f"{sequence_id}.a3m"
                    temporary_output = output.with_suffix(".a3m.tmp")
                    temporary_output.write_text(contents)
                    temporary_output.replace(output)
        except Exception:
            print(f"Retained Riboseek intermediates in {work_dir}")
            raise

    converted = 0
    for sequence_id, sequence, directory in queries.itertuples(index=False):
        a3m = directory / f"{sequence_id}.a3m"
        if not a3m.is_file():
            raise RuntimeError(f"Riboseek did not write {a3m}")
        records = read_alignment(a3m)
        if not records:
            raise RuntimeError(f"Empty alignment: {a3m}")
        query = "".join(base for base in records[0][1] if not base.islower())
        normalized = query.replace("-", "").replace("T", "U")
        if normalized != normalize_query(sequence):
            raise RuntimeError(f"Incorrect query sequence in {a3m}")
        query_record = (f">{sequence_id} {MSA_DESCRIPTION}", records[0][1])
        query_changed = records[0] != query_record
        if query_changed:
            records[0] = query_record
            temporary = a3m.with_suffix(".a3m.tmp")
            temporary.write_text(
                "".join(f"{header}\n{aligned}\n" for header, aligned in records)
            )
            temporary.replace(a3m)
        (directory / f"{sequence_id}.fa").write_text(f">{sequence_id}\n{sequence}\n")
        afa = directory / f"{sequence_id}.afa"
        if afa.is_file() and not query_changed:
            continue
        temporary = afa.with_suffix(".afa.tmp")
        subprocess.run(split(f"reformat.pl a3m fas {a3m} {temporary}"), check=True)
        temporary.replace(afa)
        converted += 1
    print(
        f"Wrote {len(missing):,} A3Ms and {converted:,} aligned FASTAs "
        f"for {len(sequences):,} unique sequences"
    )
    _link_fitness_alignments(set(queries["sequence_id"]))
    if not missing.empty:
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
