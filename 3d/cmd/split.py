#!/usr/bin/env python3

###############################################################################
# `split.py`: Splits `annotated_chain_ids.csv` into the test split.
###############################################################################

import pandas as pd

from util import Config
from util.analysis import add_seq_id, add_tm_id, prep_usalign


def debug_ecs(df: pd.DataFrame) -> None:
    """
    Prints debug information about the input DataFrame.
    """
    # --- Print info about Rfam clusters ---
    unique_rfam_names = set(df["Rfam"].unique())
    component_rfams = (
        pd.read_csv(
            "annotated_chain_ids.csv",
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )
        .groupby("Rfam Cluster")["Rfam"]
        .apply(set)
        .to_dict()
    )
    for comp_id in sorted(df["Rfam Cluster"].unique()):
        if not pd.isna(comp_id):
            rfam_names = component_rfams[comp_id]
            print(
                f"{comp_id} ({len(rfam_names)} fams): "
                + ", ".join(
                    rfam_name
                    for rfam_name in rfam_names
                    if rfam_name in unique_rfam_names
                )
            )

    # --- Print info about good/bad EC structures ---
    good_ecs = df.query(f"`Rfam E-value` < {Config.RFAM_GOOD_CUTOFF}")
    bad_ecs = df.query(f"`Rfam E-value` >= {Config.RFAM_BAD_CUTOFF} or Rfam != Rfam")
    label_to_ecs = {"good": good_ecs, "bad": bad_ecs}
    for label, ecs in label_to_ecs.items():
        n_seqs = ecs["Sequence (unmod.)"].nunique()
        n_seq_clusts = ecs["Sequence Cluster"].nunique()
        n_struct_clusts = ecs["Rfam Cluster"].nunique()
        print(
            "%d %s EC sequences belonging to %d|%d sequence|structure clusters"
            % (n_seqs, label, n_seq_clusts, n_struct_clusts)
        )
        print(f"{ecs['Rfam'].nunique()} unique {label} Rfams")


NMR_SENTINEL = 1.23456789


def get_split_candidates() -> (pd.DataFrame, pd.DataFrame):
    """
    Retrives all candidate monomer and multimer chains based on the criteria in
    `util/config.py`.  Note that TM_train and %ID_train are not yet processed.
    """
    df = pd.read_csv(
        "./annotated_chain_ids.csv",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    print(f"All RNAs: {len(df)}")
    og_cols = df.columns.tolist()

    # Drop rows that failed to download
    df = df[df["Asym. Chain ID"] != "ERROR: Failed to download"]

    # Clean up resolution column
    df["Resolution"] = (
        df["Resolution"]
        .str.split(",")
        .explode()
        .replace("N/A", NMR_SENTINEL)  # NMR structures
        .replace(".", None)
        .replace("n.s.", None)
        .astype(float)
        .groupby(level=0)
        .max()
    )

    # --- Filter RNAs ---
    # Structured
    df.columns = df.columns.str.replace("#", "no.")

    # Quality + Date cutoff
    df = df.query(
        f"Resolution <= {Config.MAX_RESOLUTION} "
        f"and `Fraction missing` <= {Config.MAX_FRAC_MISSING}"
    )
    print(f"Quality RNAs: {len(df)}")

    df = df.query(f'Published >= "{Config.TRAINING_CUTOFF}"')
    print(f"After cutoff RNAs: {len(df)}")

    # Rfam
    df.loc[:, "Rfam"] = df["Rfam"].fillna("")
    # df = df.query(
    #    f"`Rfam fraction observed` >= {Config.MIN_RFAM_OBSERVED} " f'or Rfam == ""'
    # )

    # Structured
    df = df.query(f"{Config.MIN_L} <= L")
    is_monomer = (
        f"`% covered (any polymer)` <= {Config.MAX_PCT_COVER_MONOMER} "
        f"and `Self Structured` == True "
        # NOTE(MCA): We use L here to avoid including the lengths of other
        #   polymers that will not be part of the prediction.
        f"and L <= {Config.MAX_N}"
    )
    mon_df = df.query(is_monomer)
    print(f"Monomers: {len(mon_df)}")
    is_multimer = (
        f"`% covered (any polymer)` > {Config.MAX_PCT_COVER_MONOMER} "
        # NOTE(MCA): Here, we use N because the other polymer chains will be
        #   included as part of the prediction.
        f"and N <= {Config.MAX_N}"
    )
    mul_df = df.query(is_multimer)
    print(f"Multimers: {len(mul_df)}")

    # --- Add TM ID scores to the best resolution structures of each cluster ---
    # Sort by both PDB ID and asym. chain ID to ensure strict ordering of the
    # entire dataset
    df = df.sort_values(
        by=["Resolution", "L", "PDB ID", "Asym. Chain ID"],
        ascending=[True, False, True, True],
    )

    def get_best_seqs(df):
        return df.groupby("Sequence (unmod.)").first().reset_index()

    mon_df = get_best_seqs(mon_df)
    mul_df = get_best_seqs(mul_df)
    print("After best seqs")
    print(f"{len(mon_df)=}")
    print(f"{len(mul_df)=}")
    print(f"{mul_df['PDB ID'].nunique()=}")

    return mon_df, mul_df, og_cols


def main():
    mon_df, mul_df, og_cols = get_split_candidates()
    prep_usalign()

    # --- Select up to two sequences clusters per Rfam ---
    def get_best_rfams(df):
        filtered_rows = []
        add_seq_id(df)
        add_tm_id(df)
        df = df.sort_values(
            by=[
                "AF3 TM Homolog Score",
                "AF3 Sequence Homolog %id",
                "Resolution",
                "L",
                "PDB ID",
                "Asym. Chain ID",
            ],
            ascending=[True, True, True, False, True, True],
        )
        for rfam, fam_group in df.groupby("Rfam"):
            # Keep top N sequence clusters per Rfam, or all sequence clusters
            # with no Rfam
            n = len(fam_group) if rfam == "" else Config.TOP_N
            top_n = fam_group.groupby("Sequence Cluster").first().reset_index().head(n)
            filtered_rows.append(top_n)

        return pd.concat(filtered_rows).reset_index()

    # Rank rows (lower is better)
    mon_df = get_best_rfams(mon_df)
    mul_df = get_best_rfams(mul_df)
    print("After best rfams")
    print(f"{len(mon_df)=}")
    print(f"{len(mul_df)=}")
    print(f"{mul_df['PDB ID'].nunique()=}")

    # --- Write to CSV ---
    def write_to_csv(df, fname):
        debug_ecs(df)
        df.columns = df.columns.str.replace("no.", "#")
        target_cols = og_cols
        for bl_name in Config.BASELINES.keys():
            new_col = f"{bl_name.upper()} Sequence Homolog"
            target_cols += [
                new_col,
                f"{new_col} Date",
                f"{new_col} %id",
            ]
        for bl_name in Config.BASELINES.keys():
            new_col = f"{bl_name.upper()} TM Homolog"
            target_cols += [
                new_col,
                f"{new_col} Date",
                f"{new_col} Rfam",
                f"{new_col} Score",
            ]
        df = df[target_cols]
        df["Resolution"] = df["Resolution"].replace(NMR_SENTINEL, "N/A")
        df.sort_values(by=["PDB ID", "Auth. Chain ID"]).to_csv(fname, index=False)

    write_to_csv(mon_df, Config.MONOMER_CSV)
    write_to_csv(mul_df, Config.MULTIMER_CSV)


if __name__ == "__main__":
    main()
