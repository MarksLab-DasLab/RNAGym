#!/usr/bin/env python3

###############################################################################
# `evcouplings.py`:  Code for analyzing RNA structures with EVCouplings.
###############################################################################
from __future__ import annotations

from typing import Dict, Optional, Tuple, TypeVar

from evcouplings.utils import BailoutException

from rnagym.s3d.util import ChainID, Config, PdbID
from rnagym.s3d.util.analysis import Analysis

T = TypeVar("T")
JobOutput = Optional[Tuple[PdbID, ChainID, T]]


def run_ev_couplings(data: Dict[str, str]) -> JobOutput[Exception]:
    """
    Runs the EVCoupligns pipeline for a given PDB ID and Chain ID.

    Parameters:
        data (Dict[str, str]):  A dictionary defining a "PDB ID" and "Chain ID"
          to analyze.

    Returns:
        A JobOutput[Exception] if the pipeline failed, otherwise None.
    """
    pdb_id, asym_id, auth_id = (
        data["PDB ID"],
        data["Asym. Chain ID"],
        data["Auth. Chain ID"],
    )

    try:
        analysis = Analysis(pdb_id, asym_id, auth_id)
        analysis.run_ev_couplings()
        return None
    except BailoutException as e:
        print(f"ERROR: {pdb_id}_{asym_id} failed!")
        raise e


def main():
    df = Config.load_targets()
    rows = df.to_dict(orient="records")  # Convert rows to dictionaries for easy access

    # Single-threaded (for debugging)
    _ = [run_ev_couplings(row) for row in rows if row["PDB ID"].lower()]

    # Multi-threaded (for speed)
    # with ProcessPoolExecutor() as executor:
    #    results = list(executor.map(run_ev_couplings, rows))


if __name__ == "__main__":
    main()
