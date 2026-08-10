#!/usr/bin/env python3

###############################################################################
# `merge.py`: Identify the unique PDB IDs from our five input data sources and
#   output them to `merged_pdb_ids.csv`.
###############################################################################

import json
import re
from collections import defaultdict

import pandas as pd

from rnagym.config import Config3D


def main():
    Config3D.CURATION_DIR.mkdir(parents=True, exist_ok=True)

    # Data sources
    casp15_source = "datasets/casp15/rna_targetlist.csv"
    rna_2020_source = "datasets/2020_rna/PDB/pdb_ids.csv"
    rna_puzzles_source = "datasets/rna_puzzles/rna_puzzles.csv"
    rna3dhub_full_nr_source = "datasets/rna3dhub/nrlist_3.365_all.csv"
    rna3db_parse_source = "datasets/rna3db/parse.json"

    # PDB ID regex
    pdb_id_re = re.compile(r"\b\w{4}\b")

    # CASP15
    casp15_pdb_ids = set()
    casp15_data = pd.read_csv(casp15_source, delimiter=";")
    for description in casp15_data["Description"]:
        pdb_ids = [
            pdb_id.upper()
            for pdb_id in pdb_id_re.findall(description)
            if pdb_id.upper() not in ("SARS", "HKU5")
        ]
        casp15_pdb_ids.update(pdb_ids)

    # RNA 2020
    rna_2020_pdb_ids = set()
    rna_2020_data = pd.read_csv(rna_2020_source)
    for pdb_id in rna_2020_data["PDB ID"]:
        rna_2020_pdb_ids.add(pdb_id.upper())

    # RNA Puzzles
    rna_puzzles_pdb_ids = set()
    rna_puzzles_data = pd.read_csv(rna_puzzles_source)
    for pdb_list in rna_puzzles_data["PDB ID(s)"]:
        pdb_ids = [pdb_id.upper() for pdb_id in pdb_id_re.findall(pdb_list)]
        rna_puzzles_pdb_ids.update(pdb_ids)

    # RNA3DB RNA PDBs
    with open(rna3db_parse_source, "r") as f:
        rna3db_pdb_ids = set(
            chain_id.split("_")[0].upper() for chain_id in json.loads(f.read()).keys()
        )

    # RNA 3D Hub equivalence classes
    eq_class_ids = defaultdict(set)  # Mapping from PDB ID to EQ classes
    rna3dhub_full_nrs = pd.read_csv(rna3dhub_full_nr_source)
    for _, row in rna3dhub_full_nrs.iterrows():
        # Equivalence classes
        eq_class_id = row["EC ID"]
        ranked_ife_ids = [
            chain_list.split("+") for chain_list in row["Ranked EC IFE IDs"].split(",")
        ]
        for chain_list in ranked_ife_ids:
            ec_size = len(chain_list)
            for chain in chain_list:
                pdb_id, _, chain_id = chain.split("|")
                eq_class_ids[pdb_id].add(f"{chain_id}|{eq_class_id}|{ec_size}")

    # Merge the datasets
    merged_pdb_ids = defaultdict(list)
    all_sources = [
        ("Casp15", casp15_pdb_ids),
        ("2020 RNA", rna_2020_pdb_ids),
        ("RNA Puzzles", rna_puzzles_pdb_ids),
        ("RNA3DB", rna3db_pdb_ids),
    ]

    # Populate merged_pdb_ids with each PDB ID and the sources it's found in
    for name, source in all_sources:
        print(f"{name}: {len(source)} PDB IDs")
        for pdb_id in source:
            merged_pdb_ids[pdb_id].append(name)

    # Output to merged_pdb_ids.csv with PDB ID and Source(s) columns
    with open(Config3D.MERGED_PDB_IDS_FILE, "w") as file:
        file.write("PDB ID,Source(s),Equivalence Class(es),EC size\n")
        for pdb_id, sources in sorted(merged_pdb_ids.items()):
            eq_classes = eq_class_ids[pdb_id]
            file.write(
                f'{pdb_id},"{", ".join(sources)}","{", ".join(sorted(eq_classes))}"\n'
            )


if __name__ == "__main__":
    main()
