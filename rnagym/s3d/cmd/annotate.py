#!/usr/bin/env python3

###############################################################################
# `annotate.py`: Annotates the merged PDBs with additional criteria that can be
# used for filtering.  Output is written to `annotated_chains.csv`.
###############################################################################

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import List

import pandas as pd

from rnagym.config import Config3D
from rnagym.s3d.util.analysis import assign_cluster_0
from rnagym.s3d.util.structure import StructureInfo


def process_row(row) -> List[str]:
    """
    Helper function for processing the input RNA 3D Hub CSV in parallel.
    """
    pdb_id, sources, eq_classes = row.iloc[0:3]

    if pd.isna(eq_classes):
        eq_classes = {}
    else:
        eq_classes = [eq_class.split("|") for eq_class in eq_classes.split(", ")]
        eq_classes = {
            chain_id: (eq_class, ife_size)
            for chain_id, eq_class, ife_size in eq_classes
        }

    sources = set(sources.split(", "))
    structure_info = StructureInfo.from_pdb_id(
        pdb_id, sources=sources, eq_classes=eq_classes
    )

    if structure_info is None:
        return [[pdb_id, "ERROR: Failed to download"]]

    return structure_info.get_data()


def main():
    # Write the headers
    out_fname = Config3D.ANNOTATED_CHAINS_FILE
    headers = StructureInfo.HEADERS
    with open(out_fname, "w") as file:
        file.write(f"{','.join(headers)}\n")

    # Load the data
    data_path = Config3D.MERGED_PDB_IDS_FILE
    data = pd.read_csv(data_path)

    # Write the data in 50 splits (limits total memory utilization)
    data = list(data.iterrows())
    for split in [data[i::10] for i in range(10)]:
        # Single-threaded for debugging
        # data = [process_row(row) for _, row in split if row.iloc[0] == "8TOC"]

        # Multi-threaded for speed
        with ProcessPoolExecutor() as executor:
            data = list(executor.map(process_row, (row for _, row in split)))

        data = list(filter(None, data))

        with open(out_fname, "a") as file:
            file.writelines(
                f"{','.join(chain_datum)}\n"
                for chain_data in data
                for chain_datum in chain_data
            )

    # --- Add information about leakage ---
    df = pd.read_csv(out_fname, keep_default_na=False, na_values=[], low_memory=False)
    df = df.dropna(subset="Auth. Chain ID")  # failed to download
    assign_cluster_0(df)
    rfam_to_min_pub = df.groupby("Rfam Cluster")["Published"].min().to_dict()
    seq_to_min_pub = df.groupby("Sequence Cluster")["Published"].min().to_dict()
    df.loc[:, "Earliest Rfam homolog"] = df["Rfam Cluster"].map(rfam_to_min_pub)
    df.loc[:, "Earliest sequence homolog"] = df["Sequence Cluster"].map(seq_to_min_pub)
    df = df.sort_values(by=["PDB ID", "Auth. Chain ID"])

    print(
        "%d novel structures post-2023/01/01 (excluding component 0)"
        % (df["Earliest Rfam homolog"] >= "2023-01-01").sum()
    )
    print(
        "%d novel sequences post-2023/01/01"
        % (df["Earliest sequence homolog"] >= "2023-01-01").sum()
    )

    df.to_csv(out_fname, index=False)


if __name__ == "__main__":
    main()
