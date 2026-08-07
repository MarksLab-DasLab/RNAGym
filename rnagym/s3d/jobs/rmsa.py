#!/usr/bin/env python3

###############################################################################
# `rmsa.py`: Launches slurm jobs to calculate MSAs for all RNAGym candidate
#   chains.
###############################################################################

import shlex

# import shutil
import subprocess
from pathlib import Path

import pandas as pd

from rnagym.s3d.util import Config

if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent
    mon_df = pd.read_csv(
        root_path / Config.MONOMER_CSV, keep_default_na=False, na_values=[""]
    )
    mul_df = pd.read_csv(
        root_path / Config.MULTIMER_CSV, keep_default_na=False, na_values=[""]
    )
    df = pd.concat([mon_df, mul_df], axis=0)

    # Launch rMSA jobs for each chain
    for _, row in df.iterrows():
        pdb_id = row["PDB ID"]
        chain_id = row["Asym. Chain ID"]
        seq_unmod = row["Sequence (unmod.)"]

        # Create the prefix dir, removing old runs if unsuccessful
        prefix = Path(Config.get_out_prefix(pdb_id.lower(), chain_id)) / "rMSA"
        if prefix.exists() and prefix.is_dir():
            success = prefix / "SUCCESS"
            if success.exists():
                continue
            # NOTE(MCA): rmsa can ~smartly restart from previous attempts.
            # else:
            #     shutil.rmtree(prefix)

        prefix.mkdir()

        # Change into the prefix and launch the script
        rmsa_sh = root_path / "scripts/rmsa.sh"
        sbatch_cmd = shlex.split(f"sbatch {rmsa_sh.as_posix()} {seq_unmod}")
        print(f"Launching {prefix.as_posix()}...")
        subprocess.run(sbatch_cmd, check=True, cwd=prefix)
