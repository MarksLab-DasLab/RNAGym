"""Build the shared RNAcentral and NCBI nt Riboseek databases."""

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from shlex import split

from rnagym.config import ConfigRiboseek


def set_latest(directory: Path) -> None:
    """Point the database's latest symlink to one completed version."""
    latest = directory.parent / "latest"
    temporary = latest.with_name(".latest.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(directory.name, target_is_directory=True)
    temporary.replace(latest)


def build(directory: Path, create: Callable[[Path, Path], None]) -> None:
    """Atomically build one padded Riboseek database."""
    if (directory / "SUCCESS").is_file():
        set_latest(directory)
        print(f"Skipping complete database: {directory}")
        return
    if directory.exists():
        raise RuntimeError(f"Incomplete database directory: {directory}")

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-", dir=directory.parent
    ) as temporary:
        build_dir = Path(temporary)
        database = build_dir / "riboseek"
        create(database, build_dir / "tmp")
        threads = os.environ.get("SLURM_CPUS_PER_TASK", "32")
        subprocess.run(
            split(
                f"riboseek makepaddedseqdb {database} "
                f"{database}_gpu --threads {threads}"
            ),
            check=True,
        )
        subprocess.run(split(f"riboseek rmdb {database}"), check=True)
        (build_dir / "SUCCESS").touch()
        build_dir.chmod(0o2750)
        build_dir.rename(directory)
    set_latest(directory)
    print(f"Wrote {directory}")


def create_rnacentral(database: Path, work_dir: Path) -> None:
    """Download RNAcentral and create its Riboseek database."""
    threads = os.environ.get("SLURM_CPUS_PER_TASK", "32")
    url = (
        "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/"
        f"{ConfigRiboseek.RNACENTRAL_VERSION}/sequences/rnacentral_active.fasta.gz"
    )
    fasta = database.with_suffix(".fasta.gz")
    subprocess.run(split(f"curl -L --fail --progress-bar {url} -o {fasta}"), check=True)
    subprocess.run(
        split(f"riboseek createdb {fasta} {database} --threads {threads}"),
        check=True,
    )
    fasta.unlink()


def create_nt(database: Path, work_dir: Path) -> None:
    """Download full NCBI nt and create its Riboseek database."""
    threads = os.environ.get("SLURM_CPUS_PER_TASK", "32")
    subprocess.run(
        split(f"riboseek databases NT {database} {work_dir} --threads {threads}"),
        check=True,
    )


def split_nt(directory: Path) -> None:
    """Split exceptionally long nt entries for bounded GPU memory use."""
    if ConfigRiboseek.DATABASE_NT.with_suffix(".dbtype").is_file():
        print(f"Skipping complete database: {ConfigRiboseek.DATABASE_NT}")
    else:
        threads = os.environ.get("SLURM_CPUS_PER_TASK", "32")
        with tempfile.TemporaryDirectory(prefix=".chunks-", dir=directory) as temporary:
            output = Path(temporary) / "riboseek_gpu"
            subprocess.run(
                split(
                    f"riboseek splitsequence {directory / 'riboseek_gpu'} {output} "
                    f"--max-seq-len {ConfigRiboseek.MAX_TARGET_LENGTH} "
                    f"--sequence-overlap {ConfigRiboseek.TARGET_OVERLAP} "
                    "--headers-split-mode 1 --sequence-split-mode 1 "
                    f"--create-lookup 1 --threads {threads}"
                ),
                check=True,
            )
            Path(temporary).chmod(0o2750)
            Path(temporary).rename(ConfigRiboseek.DATABASE_NT.parent)
        print(f"Wrote {ConfigRiboseek.DATABASE_NT}")

    partition_nt()


def partition_nt() -> None:
    """Repad the length-bounded nt chunks for GPU search."""
    success = ConfigRiboseek.NT_PARTITION_DIR / "SUCCESS"
    if success.is_file():
        print(f"Skipping complete database: {ConfigRiboseek.NT_PARTITION_DIR}")
        return
    if ConfigRiboseek.NT_PARTITION_DIR.exists():
        raise RuntimeError(
            f"Incomplete database directory: {ConfigRiboseek.NT_PARTITION_DIR}"
        )

    with tempfile.TemporaryDirectory(
        prefix=".partitions-", dir=ConfigRiboseek.NT_DIR
    ) as temporary:
        directory = Path(temporary)
        threads = os.environ.get("SLURM_CPUS_PER_TASK", "32")
        subprocess.run(
            split(
                f"riboseek makepaddedseqdb {ConfigRiboseek.DATABASE_NT} "
                f"{directory / 'riboseek_gpu_00'} --threads {threads}"
            ),
            check=True,
        )
        (directory / "SUCCESS").touch()
        directory.chmod(0o2750)
        directory.rename(ConfigRiboseek.NT_PARTITION_DIR)
    print(f"Wrote {ConfigRiboseek.NT_PARTITION_DIR}")


def main() -> None:
    """Build one database selected by the Slurm array index."""
    databases = (
        (ConfigRiboseek.RNACENTRAL_DIR, create_rnacentral),
        (ConfigRiboseek.NT_DIR, create_nt),
    )
    index = int(sys.argv[1])
    build(*databases[index])
    if index == 1:
        split_nt(ConfigRiboseek.NT_DIR)


if __name__ == "__main__":
    main()
