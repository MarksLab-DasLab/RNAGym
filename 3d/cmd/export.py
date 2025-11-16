#!/usr/bin/env python3

###############################################################################
# `export.py`: Neatly organizes output files for sharing with outside sources.
###############################################################################

import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

from util import Config
from util.analysis import get_rf2na_chain_ids, write_chain_minimal_pdb


def main():
    mon_df = pd.read_csv("./monomer.csv")
    mul_df = pd.read_csv("./complex.csv")

    now_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    export_dir = Path(f"./RNAGym_Export_{now_str}")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir()

    # Copy in correct PDB outputs
    def copy_pdbs(df, label):
        for _, row in df.iterrows():
            pdb_id = row["PDB ID"].lower()
            chain_id = row["Asym. Chain ID"]
            chain_key = f"{pdb_id.lower()}_{chain_id}"
            src = Config.get_minimal_pdb_file(pdb_id, chain_id)
            dst_dir = export_dir / label / chain_key
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "rcsb.pdb"
            shutil.copy2(src, dst)

    copy_pdbs(mon_df, "monomers")
    copy_pdbs(mul_df, "multimers")

    # Copy in baseline predictions
    datasets = [("monomers", mon_df), ("multimers", mul_df)]
    for label, dataset in datasets:
        for baseline in Config.BASELINES.values():
            is_multimer = label == "multimers"
            if is_multimer and baseline.mul_afa_file is None:
                continue

            for _, row in dataset.iterrows():
                pdb_id = row["PDB ID"].lower()
                chain_id = row["Asym. Chain ID"]
                chain_key = f"{pdb_id}_{chain_id}"
                out_dir, out_pdb = Config.get_bl_out_pdb(
                    baseline.name, pdb_id, chain_id, is_multimer
                )
                success_file = out_dir / "SUCCESS"

                target_chain_id = chain_id
                if success_file.exists() and out_pdb.exists():
                    if is_multimer:
                        if baseline.name == "rf2na":
                            rf2na_launch_sh = Path(out_dir / "launch.sh")
                            rf2na_chain_ids = get_rf2na_chain_ids(rf2na_launch_sh)
                            target_chain_id = rf2na_chain_ids[target_chain_id]

                        out_pdb = write_chain_minimal_pdb(
                            out_pdb, chain_id, target_chain_id
                        )

                    dst_dir = export_dir / label / chain_key
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(out_pdb, dst_dir / f"{baseline.name}.pdb")

    # --- Copy in EVCouplings predictions, MSAs and USAlign files ---
    datasets = [("monomers", mon_df), ("multimers", mul_df)]
    couplings_out = export_dir / "couplings"
    msa_dir = export_dir / "msa"
    usa_dir = export_dir / "usalign"

    # Create output directories
    for directory in (couplings_out, msa_dir, usa_dir):
        directory.mkdir()

    # Copy in per-chain files/folders
    for _, dataset in datasets:
        for _, row in dataset.iterrows():
            pdb_id = row["PDB ID"].lower()
            asym_id = row["Asym. Chain ID"]
            chain_key = f"{pdb_id}_{asym_id}"
            prefix = Config.get_out_prefix(pdb_id, asym_id)

            # EVCouplings
            couplings_dir = prefix / "couplings"
            shutil.copytree(couplings_dir, couplings_out / chain_key)

            # MSA
            couplings_afa = prefix / "rMSA" / "sequence.afa"
            shutil.copy2(couplings_afa, msa_dir / f"{chain_key}.afa")

            # USAlign output
            usa_out = Path(f"./{Config.CHAINS_DIR}/{chain_key}.out")
            shutil.copy2(usa_out, usa_dir / f"{chain_key}.usalign.out")

    # Copy in README.md describing the files
    readme = export_dir / "README.md"
    readme.write_text(
        "# RNAGym Export {}\n\n"
        "## Contents\n"
        "- `monomers/`: Monomer PDB predictions (<bl>.pdb) & baselines (rcsb.pdb)\n"
        "- `multimers/`: Multimer PDB predictions (<bl>.pdb) & baselines (rcsb.pdb)\n"
        "- `couplings/`: EVCouplings predictions\n"
        "- `msa/`: Input multiple sequence alignments (MSAs)\n"
        "- `usalign/`: USAlign output files (test-to-train TM scores)\n"
        "- `monomer.csv`: Monomer results\n"
        "- `complex.csv`: Multimer results\n".format(now_str)
    )

    # Copy in final analysis results with only interesting columns
    mon_analyzed = pd.read_csv(Config.MONOMER_ANALYZED_CSV)[Config.MON_EXPORT_COLS]
    mon_analyzed.to_csv(export_dir / "monomer.csv", index=False)
    mul_analyzed = pd.read_csv(Config.MULTIMER_ANALYZED_CSV)[Config.MUL_EXPORT_COLS]
    mul_analyzed.to_csv(export_dir / "complex.csv", index=False)


if __name__ == "__main__":
    main()
