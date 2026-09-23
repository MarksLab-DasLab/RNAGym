"""Prepare standard nucleotide databases from the Riboseek source releases."""

import json
import os
import subprocess
import sys
from pathlib import Path
from shlex import quote, split

from rnagym.config import ConfigMMseqs, ConfigRiboseek


def run(command: str) -> None:
    """Run a command, retaining its arguments in the job log."""
    print(command, flush=True)
    subprocess.run(split(command), check=True)


def link_nt(source: Path, output: Path) -> None:
    """Expose the existing ASCII nt chunks as a standard nucleotide database.

    The chunk index addresses the original nucleotide text, not the padded
    dinucleotide database used by Riboseek's GPU search. Only the new database
    receives a standard MMseqs2 nucleotide type marker.
    """
    with source.open("rb") as handle, source.with_suffix(".index").open() as index:
        for _, line in zip(range(1024), index):
            _, offset, length = map(int, line.split())
            handle.seek(offset)
            sequence = handle.read(length - 2)
            if not sequence or set(sequence.upper()) - set(b"ACGTURYSWKMBDHVNX"):
                raise ValueError(f"Expected raw nucleotide text in {source}")
    for suffix in ("", ".index", ".lookup", "_h", "_h.index", "_h.dbtype"):
        target = Path(f"{source}{suffix}").resolve(strict=True)
        link = Path(f"{output}{suffix}")
        if link.is_symlink() and link.resolve() == target:
            continue
        link.symlink_to(target)
    output.with_suffix(".dbtype").write_bytes((1).to_bytes(4, sys.byteorder))


def prepare(index: int) -> None:
    """Build one source database without changing the Riboseek databases."""
    output = ConfigMMseqs.DATABASES[index]
    directory = output.parent
    marker = directory / "SUCCESS.json"
    if marker.is_file():
        if output.with_suffix(".dbtype").read_bytes() != (1).to_bytes(4, sys.byteorder):
            raise ValueError(f"Expected a standard nucleotide database: {output}")
        print(f"Database is complete: {output}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    if index == 0:
        version = ConfigRiboseek.RNACENTRAL_VERSION
        url = (
            "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/"
            f"{version}/sequences/rnacentral_active.fasta.gz"
        )
        fasta = directory / "rnacentral_active.fasta.gz"
        downloaded = fasta.with_suffix(".downloaded")
        if not downloaded.is_file():
            run(
                f"curl --fail --location --retry 5 --continue-at - "
                f"{quote(url)} --output {quote(str(fasta))}"
            )
            run(f"gzip --test {quote(str(fasta))}")
            downloaded.touch()
        # createdb is inexpensive relative to the download and is safe to repeat
        run(
            f"mmseqs createdb {quote(str(fasta))} {quote(str(output))} "
            "--dbtype 2 --shuffle 0"
        )
        provenance = {"release": version, "url": url}
    else:
        source = ConfigRiboseek.DATABASE_NT
        link_nt(source, output)
        provenance = {
            "release": ConfigRiboseek.NT_VERSION,
            "source": str(source),
            "chunk_length": ConfigRiboseek.MAX_TARGET_LENGTH,
            "chunk_overlap": ConfigRiboseek.TARGET_OVERLAP,
        }
    marker.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Database is complete: {output}", flush=True)


if __name__ == "__main__":
    prepare(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
