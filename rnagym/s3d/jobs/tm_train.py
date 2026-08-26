#!/usr/bin/env python3

###############################################################################
# `tm_train.py`: Launches slurm jobs to calculate TM_train for all RNAGym
#   candidate chains.
###############################################################################

import itertools
import shlex
import subprocess
from rnagym.s3d.cmd.split import get_split_candidates
from pathlib import Path
from typing import List

from rnagym.s3d.util import ChainID, Config
from rnagym.s3d.util.analysis import prep_usalign


def get_split_candidate_chain_ids(verbose=False) -> (List[ChainID], List[ChainID]):
    """
    Retrives all candidate monomer and multimer chain IDs based on the criteria
    in `util/config.py`.
    """

    def get_chain_ids(df):
        return (df["PDB ID"].str.lower() + "_" + df["Asym. Chain ID"]).tolist()

    mon_df, mul_df, _ = get_split_candidates()
    mon_ids, mul_ids = get_chain_ids(mon_df), get_chain_ids(mul_df)

    # Print, if requested:
    if verbose:
        print("\n".join(mon_ids))
        print("\n".join(mul_ids))

    return mon_ids, mul_ids


if __name__ == "__main__":
    mon_ids, mul_ids = get_split_candidate_chain_ids()

    # Prepare USAlign then run individual jobs across multiple CPUs
    prep_usalign()
    for chain_id in itertools.chain(mon_ids, mul_ids):
        pdb_id, asym_id = chain_id.split("_")

        # Run usalign only if the cached out file is absent or if it is present
        # but indicates that the previous run failed
        cached_out_file = Path(f"{Config.CHAINS_DIR}/{pdb_id.lower()}_{asym_id}.out")
        if cached_out_file.exists():
            with open(cached_out_file, "r") as f:
                usa_results = f.read()
        else:
            usa_results = ""

        _, _, last_line = usa_results.rstrip().rpartition("\n")
        if (
            usa_results == ""
            or last_line.startswith("#Total CPU time is  0.00 seconds")
            or not last_line.startswith("#Total CPU time is")
        ):
            sbatch_cmd = shlex.split(
                f"sbatch scripts/us_align.sh {pdb_id.upper()} {asym_id}"
            )
            subprocess.run(sbatch_cmd, check=True)
