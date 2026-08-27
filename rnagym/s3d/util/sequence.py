#!/usr/bin/env python3

###############################################################################
# `sequence.py`:  Helper classes for working with sequences and sequence
#   alignments
###############################################################################
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

import evcouplings.align.alignment as Alignment
import pandas as pd
from rna3db.tabular import TabularOutput

from rnagym.s3d.util import AccessionID, ChainID, Config, PdbID, Sequence


@cache
def rfam_families() -> pd.DataFrame:
    """Load family sizes and covariance-model lengths from Rfam."""
    # Columns follow Rfam's official family table schema
    return pd.read_csv(
        Config.RFAM_FAMILY_TABLE,
        sep="\t",
        header=None,
        encoding="latin-1",
        usecols=[0, 15, 29],
        names=["accession", "num_full", "model_length"],
        index_col="accession",
    )


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

    # Instance variables
    __slots__ = ("data_frame", "__fam_hits")

    class Source(Enum):
        """
        Valid FamHits sources.  Currently just Rfam and Pfam.
        """

        RFAM = auto()

    # Static variables
    QueryID = str

    __cache: Dict[Sequence, FamHits] = {}

    # Instance variables
    data_frame: Optional[pd.DataFrame]
    __fam_hits: List[FamHit]

    def __init__(self, data_frame: pd.DataFrame, source: Source):
        """
        Initializes a FamHits object using the input data frame based on
        the data source.
        """
        self.data_frame = data_frame
        self.__fam_hits = []

        if data_frame is None:
            return

        for _, row in data_frame.iterrows():
            if source == FamHits.Source.RFAM:
                self.__fam_hits.append(
                    FamHit(
                        source=FamHits.Source.RFAM,
                        name=str(row["target_name"]),
                        accession=AccessionID(row["target_accession"]),
                        score=float(row["score"]),
                        model_len=int(row["mdl_len"]),
                        seq_len=int(row["seq_len"]),
                        e_value=float(row["e_value"]),
                    )
                )
            else:
                raise ValueError(f"Unknown source {source}")

    @staticmethod
    def from_fam(
        pdb_id: PdbID,
        asym_chain_id: ChainID,
        auth_chain_id: ChainID,
        fasta_file: NamedTemporaryFile,
        fam_source: FamHits.Source,
    ) -> FamHits:
        """
        Constructs a Pfam FamHits for the input sequence as a pandas DataFrame.
        Returns the FamHits, and writes outputs to
        `data/3d/cache/{pdb_id}/{chain_id}/`.
        """
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
                pdb_id, asym_chain_id, auth_chain_id, fasta_file
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
    ) -> Optional[pd.DataFrame]:
        """
        Constructs an Rfam FamHits for the input sequence as a pandas
        DataFrame. Returns the hits, and writes outputs to
        `data/3d/cache/{pdb_id}/{chain_id}/`.
        """
        # Get the sequence
        _, sequence = next(Alignment.read_fasta(fasta_file))
        seq_len = len(sequence)

        # Conduct the cmscan
        query_id = f"{pdb_id.lower()}_{auth_chain_id}"
        tbl = None
        if query_id in Config.RNA3DB_HITS:
            # Use the RNA3DB cached hit
            tbl = Config.RNA3DB_HITS[query_id]
        else:
            # Conduct our own Rfam search
            out_prefix = Config.get_out_prefix(pdb_id=pdb_id, chain_id=asym_chain_id)
            out_table = f"{out_prefix}/rfam_table.txt"
            cmscan_cmd = shlex.split(
                f"cmscan --mid --cpu 1 --fmt 1 -o {out_prefix}/cmscan.out "
                f"--tblout {out_table} {Config.RFAM_CM} {fasta_file.name}"
            )

            try:
                os.makedirs(out_prefix, exist_ok=True)
                result = subprocess.run(cmscan_cmd, text=True, check=True)

                if result.stderr:
                    raise subprocess.CalledProcessError(result.stderr)

                tbl = TabularOutput(out_table)
            except subprocess.CalledProcessError as e:
                print(f"Unable to retrieve rfam hits for {pdb_id}_{asym_chain_id}: {e}")
                print(f"{e.stderr=}")
                fasta_file.seek(0)
                contents = fasta_file.read()
                fasta_file.seek(0)
                print(f"Fasta file contents are: {contents}")
            except pd.errors.EmptyDataError:
                print(f"Identified no Rfam hits for {pdb_id}_{asym_chain_id}")

        hits = None
        if tbl is not None:
            hits = pd.DataFrame(tbl.hits, columns=TabularOutput.TBL_ROW_TYPES)
            hits["mdl_len"] = hits["target_accession"].apply(
                FamHits.get_rfam_model_length
            )
            hits["seq_len"] = seq_len

        return FamHits(hits, FamHits.Source.RFAM)

    @staticmethod
    def get_rfam_length(accession: AccessionID) -> int:
        """Return the number of full Rfam family members."""
        return int(rfam_families().loc[accession, "num_full"])

    @staticmethod
    def get_rfam_model_length(accession: AccessionID) -> int:
        """Return the covariance-model length for one Rfam family."""
        return int(rfam_families().loc[accession, "model_length"])

    def __getitem__(self, index) -> Optional[FamHit]:
        """
        Retrieves a given hit by index. Indices are sorted by score, descending
        from 0.
        """
        return self.__fam_hits[index] if (index < len(self.__fam_hits)) else None
