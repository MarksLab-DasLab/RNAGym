#!/usr/bin/env python3

###############################################################################
# `predict.py`: Run baseline predictions for all chains.
###############################################################################

import argparse
import pandas as pd
import shlex
import shutil
import subprocess
from pathlib import Path

from util import Config
from util.analysis import prep_af3, prep_nufold, prep_rf2na, prep_rhofold, prep_trRNA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline predictions for all chains"
    )
    parser.add_argument(
        "baseline",
        choices=["ALL", "af3", "rf2na", "rho", "nu", "trRNA"],
        help="The baseline to run",
    )

    # Create a mutually exclusive group for monomer/multimer
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only_monomer", action="store_true", help="Monomers only")
    group.add_argument("--only_multimer", action="store_true", help="Multimers only")

    return parser.parse_args()


BL_PREP_FNS = {
    "af3": prep_af3,
    "nu": prep_nufold,
    "rf2na": prep_rf2na,
    "rho": prep_rhofold,
    "trRNA": prep_trRNA,
}

mon_chain_keys = {
    f"{pdb_id.lower()}_{asym_id}"
    for _, pdb_id, asym_id in pd.read_csv(Config.MONOMER_CSV)[
        ["PDB ID", "Asym. Chain ID"]
    ].itertuples()
}
mul_pdb_ids = {
    f"{pdb_id.lower()}" for pdb_id in pd.read_csv(Config.MULTIMER_CSV)["PDB ID"]
}


if __name__ == "__main__":
    args = parse_args()

    if args.baseline == "ALL":
        baselines = list(Config.BASELINES.keys())
    else:
        baselines = [args.baseline]

    for baseline in baselines:
        bl_cfg = Config.BASELINES[baseline]
        bl_dir, bl_out, bl_job, bl_prep_fn = (
            bl_cfg.install_dir,
            bl_cfg.out_dir,
            bl_cfg.job_sh,
            BL_PREP_FNS[bl_cfg.name],
        )
        bl_pred_path = Path(bl_out).resolve()
        bl_path = Path(bl_dir).resolve()
        bl_job_path = Path(bl_job).resolve()

        if args.only_monomer:
            config_paths = [bl_pred_path / "monomers"]
        elif args.only_multimer:
            config_paths = [bl_pred_path / "multimers"]
        else:
            config_paths = [bl_pred_path / "monomers", bl_pred_path / "multimers"]

        if args.only_multimer and bl_cfg.mul_afa_file is None:
            raise ValueError(f"{bl_cfg} does not support multimer predictions")

        # Iterator for all prediction directories
        def iter_pred_dirs():
            for path in config_paths:
                if path.exists() and path.is_dir():
                    for pred_dir in path.iterdir():
                        chain_dir = path.name  # monomer or multimer
                        name = pred_dir.name

                        # Only launch jobs for chain keys or PDB IDs currently
                        # found in `monomer.csv` or `complex.csv`
                        if chain_dir == "monomers" and name not in mon_chain_keys:
                            continue
                        if chain_dir == "multimers" and name not in mul_pdb_ids:
                            continue

                        if pred_dir.is_dir():
                            yield pred_dir.resolve()

        # Remove old runs if not successful
        for pred_dir in iter_pred_dirs():
            pred_dir = pred_dir.resolve()

            # Remove all files if previous run was not successful
            if not (pred_dir / "SUCCESS").exists():
                for item in pred_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()

        # Write new run configurations
        bl_prep_fn()

        # Launch predictions where needed
        for pred_dir in iter_pred_dirs():
            if (pred_dir / "SUCCESS").exists():
                continue

            print(f"Launching {str(pred_dir)}")
            sbatch_cmd = f"sbatch '{bl_job_path}' '{bl_path}'"
            subprocess.run(shlex.split(sbatch_cmd), check=True, cwd=pred_dir)
