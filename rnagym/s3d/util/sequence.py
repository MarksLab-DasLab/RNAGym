#!/usr/bin/env python3

###############################################################################
# `sequence.py`:  Helper classes for working with sequences and sequence
#   alignments
###############################################################################
from __future__ import annotations

import json
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from tempfile import NamedTemporaryFile

import polars as pl
from rna3db.tabular import TabularOutput, read_tbls_from_dir

from rnagym.config import Config3D
from rnagym.s3d.util import AccessionID, ChainID, PdbID, Sequence


@cache
def rfam_families() -> dict[str, tuple[int, int]]:
    """Load family sizes and covariance-model lengths from Rfam."""
    # Columns follow Rfam's official family table schema
    families = pl.read_csv(
        Config3D.RFAM_FAMILY_TABLE,
        separator="\t",
        has_header=False,
        columns=[0, 15, 29],
        new_columns=["accession", "n_full", "model_length"],
        encoding="utf8-lossy",
    )
    return {
        accession: (n_full, model_length)
        for accession, n_full, model_length in families.iter_rows()
    }


@cache
def rna3db_hits() -> dict[str, TabularOutput]:
    """Load the released RNA3DB Rfam hits on first use."""
    hits = defaultdict(list)
    for hit in read_tbls_from_dir(Config3D.RNA3DB_CMSCAN_DIR):
        hits[hit.query_name].append(hit)
    queries = json.loads(Config3D.RNA3DB_PARSE_FILE.read_text())
    return {query: TabularOutput(hits=hits[query]) for query in queries}


@dataclass
class FamHit:
    """
    Class representing a single family hit from Rfam.
    """

    source: FamHits.Source
    name: str  # name of the matching family
    accession: AccessionID  # accession code
    score: float  # bit score
    model_len: int  # total length of the Rfam model sequence
    seq_len: int  # length of the query sequence
    e_value: float  # E-value

    @property
    def n_over_l(self):
        """
        Return the classic N/L ratio of an MSA: the # of sequences in the
        alignment divided by the length of the sequence.
        """
        if self.source == FamHits.Source.RFAM:
            return FamHits.get_rfam_length(self.accession) / self.seq_len
        else:
            raise NotImplementedError(
                f"n_over_l not yet implemented for {self.source.name}"
            )


class FamHits:
    """
    Class for creating and working with Pfam/Rfam hits from a given sequence.
    """

    __slots__ = ("__fam_hits",)

    class Source(Enum):
        """
        Valid FamHits sources.  Currently just Rfam and Pfam.
        """

        RFAM = auto()

    # Static variables
    QueryID = str

    __cache: dict[Sequence, FamHits] = {}

    __fam_hits: list[FamHit]

    def __init__(self, hits: list[TabularOutput.Hit], source: Source, seq_len: int):
        """Create family hits from parsed Infernal output."""
        if source != FamHits.Source.RFAM:
            raise ValueError(f"Unknown source {source}")
        self.__fam_hits = [
            FamHit(
                source=source,
                name=hit.target_name,
                accession=AccessionID(hit.target_accession),
                score=hit.score,
                model_len=FamHits.get_rfam_model_length(hit.target_accession),
                seq_len=seq_len,
                e_value=hit.e_value,
            )
            for hit in hits
        ]

    @staticmethod
    def from_fam(
        pdb_id: PdbID,
        asym_chain_id: ChainID,
        auth_chain_id: ChainID,
        fasta_file: NamedTemporaryFile,
        fam_source: FamHits.Source,
    ) -> FamHits:
        """Return cached or newly computed family hits for one sequence."""
        # Extract the sequence
        sequence_lines = fasta_file.read().splitlines()[1:]
        fasta_file.seek(0)
        sequence = "".join(line.strip() for line in sequence_lines)

        # Return the cached FamHits if it exists
        if sequence in FamHits.__cache:
            return FamHits.__cache[sequence]

        # Otherwise, calculate the fam hits, store it in cache, and return it
        if fam_source == FamHits.Source.RFAM:
            fam_hits = FamHits.__from_rfam(
                pdb_id, asym_chain_id, auth_chain_id, fasta_file, len(sequence)
            )
        else:
            raise ValueError(f"Unknown source {fam_source}")

        FamHits.__cache[sequence] = fam_hits
        return fam_hits

    @staticmethod
    def __from_rfam(
        pdb_id: PdbID,
        asym_chain_id: ChainID,
        auth_chain_id: ChainID,
        fasta_file: NamedTemporaryFile,
        seq_len: int,
    ) -> FamHits:
        """Load released RNA3DB hits or scan one sequence against Rfam."""
        query_id = f"{pdb_id.lower()}_{auth_chain_id}"
        tbl = None
        cached_hits = rna3db_hits()
        if query_id in cached_hits:
            # Use the RNA3DB cached hit
            tbl = cached_hits[query_id]
        else:
            # Conduct our own Rfam search
            out_prefix = Config3D.chain_dir(pdb_id, asym_chain_id)
            out_table = out_prefix / "rfam_table.txt"
            cmscan_cmd = shlex.split(
                f"cmscan --mid --cpu 1 --fmt 1 -o {out_prefix}/cmscan.out "
                f"--tblout {out_table} {Config3D.RFAM_CM} {fasta_file.name}"
            )

            try:
                out_prefix.mkdir(parents=True, exist_ok=True)
                subprocess.run(cmscan_cmd, check=True)
                tbl = TabularOutput(out_table)
            except subprocess.CalledProcessError as e:
                print(f"Unable to retrieve rfam hits for {pdb_id}_{asym_chain_id}: {e}")
                fasta_file.seek(0)
                contents = fasta_file.read()
                fasta_file.seek(0)
                print(f"Fasta file contents are: {contents}")
        return FamHits([] if tbl is None else tbl.hits, FamHits.Source.RFAM, seq_len)

    @staticmethod
    def get_rfam_length(accession: AccessionID) -> int:
        """Return the number of full Rfam family members."""
        return rfam_families()[accession][0]

    @staticmethod
    def get_rfam_model_length(accession: AccessionID) -> int:
        """Return the covariance-model length for one Rfam family."""
        return rfam_families()[accession][1]

    def __getitem__(self, index) -> FamHit | None:
        """
        Retrieves a given hit by index. Indices are sorted by score, descending
        from 0.
        """
        return self.__fam_hits[index] if (index < len(self.__fam_hits)) else None
