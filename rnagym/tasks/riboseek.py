"""Generate shared Riboseek MSAs for 3D and fitness sequences."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from shlex import split

import pandas as pd

from rnagym.config import Config2D, Config3D, ConfigFitness, ConfigRiboseek
from rnagym.sequences import fitness_sequences, load_registry

MSA_DESCRIPTION = "Riboseek CM"


def read_a3m(path: Path) -> list[tuple[str, str]]:
    """Read A3M records while joining wrapped sequence lines."""
    records: list[list[str]] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            records.append([line, ""])
        else:
            records[-1][1] += line
    return [(header, sequence) for header, sequence in records]


def is_complete(path: Path, sequence_id: str) -> bool:
    """Check that an A3M was generated with covariance model realignment."""
    if not path.is_file():
        return False
    with path.open() as handle:
        return handle.readline().rstrip() == f">{sequence_id} {MSA_DESCRIPTION}"


def main() -> None:
    """Search RNAcentral and nt for one query shard, then write merged MSAs."""
    task_id, task_count = map(int, sys.argv[1:])
    threads = os.environ.get("SLURM_CPUS_PER_TASK", "32")
    databases = (ConfigRiboseek.DATABASE_RNACENTRAL, *ConfigRiboseek.DATABASE_NT_PARTS)
    if not ConfigRiboseek.DATABASE_NT_PARTS or any(
        not database.with_suffix(".dbtype").is_file() for database in databases
    ):
        raise FileNotFoundError("Run 'pixi run riboseek-db' first")
    targets = (
        pd.read_parquet(Config3D.TARGET_FILE)[["sequence_id", "Sequence (unmod.)"]]
        .rename(columns={"Sequence (unmod.)": "sequence"})
        .drop_duplicates()
        .assign(msa_dir=Config3D.MSA_DIR)
    )
    fitness_source = fitness_sequences()
    fitness = fitness_source.join(load_registry(Config2D.SEQUENCE_FILE), on="sequence")
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
    queries = queries[queries["sequence_id"].isin(sequences["sequence_id"])]

    for directory in queries["msa_dir"].unique():
        directory.mkdir(parents=True, exist_ok=True)
    missing_ids = {
        row.sequence_id
        for row in queries.itertuples(index=False)
        if not is_complete(row.msa_dir / f"{row.sequence_id}.a3m", row.sequence_id)
    }
    missing = sequences[sequences["sequence_id"].isin(missing_ids)]
    if not missing.empty:
        with tempfile.TemporaryDirectory(prefix="rnagym-riboseek-") as temporary:
            work_dir = Path(temporary)
            fasta = work_dir / "queries.fasta"
            fasta.write_text(
                "".join(
                    f">{sequence_id}\n{sequence}\n"
                    for sequence_id, sequence in missing.itertuples(index=False)
                )
            )
            query_db = work_dir / "queries"
            subprocess.run(
                split(f"riboseek createdb {fasta} {query_db} --threads {threads}"),
                check=True,
            )

            databases = []
            alignment_dbs = []
            for name, search_databases, strand in (
                (
                    "rnacentral",
                    (ConfigRiboseek.DATABASE_RNACENTRAL,),
                    1,
                ),
                (
                    "nt",
                    ConfigRiboseek.DATABASE_NT_PARTS,
                    2,
                ),
            ):
                for index, search_database in enumerate(search_databases):
                    alignment_db = work_dir / f"{name}_{index}_alignments"
                    # Stream database pages once during the GPU scan
                    command = (
                        f"riboseek search {query_db} {search_database} {alignment_db} "
                        f"{work_dir / f'{name}_{index}_tmp'} -a --gpu 1 "
                        "--db-load-mode 2 --max-seqs 10000 "
                        f"--strand {strand} --threads {threads}"
                    )
                    subprocess.run(split(command), check=True)
                    databases.append(search_database)
                    alignment_dbs.append(alignment_db)

            # Build a covariance model from all hits and use it to realign them
            cm = work_dir / "cm"
            subprocess.run(
                split(
                    f"riboseek cmbuild {query_db} {','.join(map(str, databases))} "
                    f"{','.join(map(str, alignment_dbs))} {cm} --threads {threads}"
                ),
                check=True,
            )
            cm_alignments = work_dir / "cm_alignments"
            subprocess.run(
                split(
                    f"riboseek cmsearch {cm} {cm}_target_merged "
                    f"{cm}_result_merged {cm_alignments} --threads {threads}"
                ),
                check=True,
            )
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
                contents = (unpacked / f"{key}.a3m").read_text()
                for directory in queries.loc[
                    queries["sequence_id"] == sequence_id, "msa_dir"
                ]:
                    output = directory / f"{sequence_id}.a3m"
                    temporary_output = output.with_suffix(".a3m.tmp")
                    temporary_output.write_text(contents)
                    temporary_output.replace(output)

    converted = 0
    for sequence_id, sequence, directory in queries.itertuples(index=False):
        a3m = directory / f"{sequence_id}.a3m"
        if not a3m.is_file():
            raise RuntimeError(f"Riboseek did not write {a3m}")
        records = read_a3m(a3m)
        if not records:
            raise RuntimeError(f"Empty alignment: {a3m}")
        query = "".join(base for base in records[0][1] if not base.islower())
        normalized = query.replace("-", "").replace("T", "U")
        if normalized != sequence.replace("I", "X").replace("N", "X"):
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


if __name__ == "__main__":
    main()
