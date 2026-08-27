"""Annotate candidate PDB RNA chains for filtering."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from csv import writer

import pandas as pd

from rnagym.config import Config3D
from rnagym.s3d.util.analysis import assign_cluster_0
from rnagym.s3d.util.structure import StructureInfo


def process_pdb(pdb_id: str) -> list[list[str]]:
    """Annotate every RNA chain in one PDB entry."""
    structure_info = StructureInfo.from_pdb_id(pdb_id)

    if structure_info is None:
        return []

    return structure_info.get_data()


def main() -> None:
    """Write the annotated RNA chain table."""
    output = Config3D.ANNOTATED_CHAINS_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    headers = StructureInfo.HEADERS
    with temporary.open("w", newline="") as file:
        writer(file).writerow(headers)

    # Process in 10 batches to limit memory use
    chains = json.loads(Config3D.RNA3DB_PARSE_FILE.read_text())
    pdb_ids = sorted({chain.split("_", 1)[0].upper() for chain in chains})
    for batch in [pdb_ids[i::10] for i in range(10)]:
        with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as executor:
            results = filter(None, executor.map(process_pdb, batch))
        with temporary.open("a", newline="") as file:
            writer(file).writerows(
                chain_datum for chain_data in results for chain_datum in chain_data
            )

    df = pd.read_csv(temporary, keep_default_na=False, na_values=[], low_memory=False)
    assign_cluster_0(df)
    rfam_to_min_pub = df.groupby("Rfam Cluster")["Published"].min().to_dict()
    seq_to_min_pub = df.groupby("Sequence Cluster")["Published"].min().to_dict()
    df.loc[:, "Earliest Rfam homolog"] = df["Rfam Cluster"].map(rfam_to_min_pub)
    df.loc[:, "Earliest sequence homolog"] = df["Sequence Cluster"].map(seq_to_min_pub)
    df = df.sort_values(by=["PDB ID", "Auth. Chain ID"])

    cutoff = Config3D.TARGET_CUTOFF
    print(
        f"{(df['Earliest Rfam homolog'] > cutoff).sum()} novel structures "
        f"after {cutoff} (excluding component 0)"
    )
    print(
        f"{(df['Earliest sequence homolog'] > cutoff).sum()} novel sequences "
        f"after {cutoff}"
    )

    df.to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"Wrote {len(df):,} chains to {output}")


if __name__ == "__main__":
    main()
