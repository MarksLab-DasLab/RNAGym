#!/usr/bin/env python3

###############################################################################
# `dataset_stats.py`:  Output information about monomer and multimer datasets.
###############################################################################

import pandas as pd
from rnagym.s3d.util import Config

mon_df = Config.load_targets("monomer")
mul_df = Config.load_targets("multimer")
cat_df = pd.concat([mon_df, mul_df])
dfs = {"monomers": mon_df, "multimers": mul_df, "total": cat_df}

headers = ["dataset", "n_chains", "length", "resolution"]
rows = []
for label, df in dfs.items():
    # 1. Get number of RNA chains
    # 2. Get number of distinct Rfams
    # 3. Resolution (avg.)
    # 3. Length (avg.)
    fields = [label]
    fields.append(len(df))
    fields.append(df["Rfam"].nunique())
    fields.append(df["Resolution"].mean(skipna=True))
    fields.append(f"{df['L'].min()}-{df['L'].max()}")
    rows.append(fields)

print(", ".join(headers))
for row in rows:
    print(", ".join(map(str, row)))

# Print Rfamless
print("")
print("Info on Rfamless chains")
for label, df in dfs.items():
    df = df[(df["Rfam E-value"] >= 1.0) | df["Rfam"].isna()]
    n_structured = df["Self Structured"].sum()
    avg_L = df["L"].mean()
    print(
        f"{label} has {len(df)} Rfamless chains (avg. L {avg_L:.1f}, "
        f"{n_structured / len(df) * 100.0:.2f}% self-structured)"
    )
