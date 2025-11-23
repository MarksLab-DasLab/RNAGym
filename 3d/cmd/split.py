#!/usr/bin/env python3

###############################################################################
# `split.py`: Splits `annotated_chain_ids.csv` into the test split.
###############################################################################

import pandas as pd

from util import Config
from util.analysis import add_seq_id, add_tm_id, prep_usalign


NMR_SENTINEL = 1.23456789


def debug_df(df: pd.DataFrame) -> None:
    """
    Prints debug information about the input DataFrame.
    """
    # Count Rfam hits (E-value < 1)
    rfam_hits = df.query("`Rfam E-value` < 1")["Rfam"].nunique()
    print(f"{rfam_hits} unique Rfam hits (E-value < 1)")

    # Count unique Rfam signatures (any detected Rfam)
    df["Rfam E-value"] = pd.to_numeric(df["Rfam E-value"], errors="coerce")
    count = len(df.query('Rfam == "" or `Rfam E-value` > 10'))
    print(f"RNAs with Rfam E-value >10.0: {count}")

    # Average length
    avg_length = df["L"].mean()
    print(f"Average length: {avg_length:.0f} nt")

    # Average resolution (excluding NMR and N/A)
    res_numeric = df[(df["Resolution"] != NMR_SENTINEL) & (df["Resolution"] != "N/A")][
        "Resolution"
    ]
    avg_res = res_numeric.mean()
    print(f"Average resolution: {avg_res:.2f} Å")


def get_split_candidates() -> (pd.DataFrame, pd.DataFrame, list):
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

    # --- Create training set ---
    print("\n--- Creating training set ---")
    test_df = pd.concat([mon_df, mul_df])
    test_ids = set(zip(test_df["PDB ID"], test_df["Asym. Chain ID"]))

    # Reload and apply quality filters (but not date filter)
    all_chains_df = pd.read_csv(
        "./annotated_chain_ids.csv",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    all_chains_df = all_chains_df[
        all_chains_df["Asym. Chain ID"] != "ERROR: Failed to download"
    ]
    all_chains_df["Resolution"] = (
        all_chains_df["Resolution"]
        .str.split(",")
        .explode()
        .replace("N/A", NMR_SENTINEL)
        .replace(".", None)
        .replace("n.s.", None)
        .astype(float)
        .groupby(level=0)
        .max()
    )

    # Apply quality filters and date filter for TRAINING (before cutoff)
    train_df = all_chains_df.query(
        f"Resolution <= {Config.MAX_RESOLUTION} "
        f"and `Fraction missing` <= {Config.MAX_FRAC_MISSING} "
        f"and L >= {Config.MIN_L} "
        f'and Published < "{Config.TRAINING_CUTOFF}"'
    )

    # Remove test set chains and deduplicate by sequence
    id_tuples = zip(train_df["PDB ID"], train_df["Asym. Chain ID"])
    train_df = train_df[[i not in test_ids for i in id_tuples]].copy()
    train_df = (
        train_df.sort_values(
            by=["Resolution", "L", "PDB ID", "Asym. Chain ID"],
            ascending=[True, False, True, True],
        )
        .groupby("Sequence (unmod.)")
        .first()
        .reset_index()
    )

    # Count monomers and multimers
    train_mon = train_df[
        train_df["% covered (any polymer)"] <= Config.MAX_PCT_COVER_MONOMER
    ]
    train_mul = train_df[
        train_df["% covered (any polymer)"] > Config.MAX_PCT_COVER_MONOMER
    ]
    print(f"Training set: {len(train_mon)} monomers, {len(train_mul)} multimers")

    # Write to CSV
    train_df["Resolution"] = train_df["Resolution"].replace(NMR_SENTINEL, "N/A")
    train_df.sort_values(by=["PDB ID", "Auth. Chain ID"]).to_csv(
        "train.csv", index=False
    )
    print(f"{len(train_df)} training chains written to train.csv")
    debug_df(train_df)
    print("")

    # Debug the RNAGym test dataset
    print("--- Debug info for test dataset ---")
    debug_df(test_df)
    print("")

    # Debug the full RNAGym dataset
    print("--- Debug info for full dataset ---")
    all_data = pd.concat([train_df, test_df])
    debug_df(all_data)
    print("")

    # Calculate TM_train between test and train using AF3 TM_train, since it is
    # the most recent model and therefore sets the training set date cutoff.
    # Note this is slightly approximate, as our train set was filtered for high
    # quality structures only.
    print("\n--- Approximate test-to-train homology ---")
    test_tm = pd.to_numeric(test_df["AF3 TM Homolog Score"], errors="coerce").dropna()
    print(
        f"min={test_tm.min():.3f}, max={test_tm.max():.3f}, "
        f"avg={test_tm.mean():.3f}, median={test_tm.median():.3f}"
    )
    print("")

    # --- Write to CSV ---
    def write_to_csv(df, fname):
        print(f"--- Writing {fname}... ---")
        debug_df(df)
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
        print("")

    write_to_csv(mon_df, Config.MONOMER_CSV)
    write_to_csv(mul_df, Config.MULTIMER_CSV)


if __name__ == "__main__":
    main()
