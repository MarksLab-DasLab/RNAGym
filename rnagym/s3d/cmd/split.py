#!/usr/bin/env python3

###############################################################################
# `split.py`: Splits `annotated_chains.csv` into the test split.
###############################################################################

import pandas as pd

from rnagym.config import Config3D
from rnagym.s3d.util import Config as Pipeline
from rnagym.s3d.util.analysis import add_seq_id, add_tm_id, prep_usalign

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
        Config3D.ANNOTATED_CHAINS_FILE,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    print(f"All RNAs: {len(df)}")
    og_cols = df.columns.tolist()
    quality = df.apply(Config3D.passes_quality, axis=1)
    monomers = df.apply(Config3D.is_monomer, axis=1)

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
    df = df[quality]
    print(f"Quality RNAs: {len(df)}")

    df = df.query(f'Published >= "{Pipeline.TRAINING_CUTOFF}"')
    print(f"After cutoff RNAs: {len(df)}")

    # Rfam
    df.loc[:, "Rfam"] = df["Rfam"].fillna("")
    # df = df.query(
    #    f"`Rfam fraction observed` >= {Config.MIN_RFAM_OBSERVED} " f'or Rfam == ""'
    # )

    # Structured
    df = df.query(f"{Config3D.MIN_LENGTH} <= L")
    mon_df = df[monomers.loc[df.index]]
    print(f"Monomers: {len(mon_df)}")
    is_multimer = (
        f"`% covered (any polymer)` > {Config3D.MAX_POLYMER_COVERAGE} "
        # NOTE(MCA): Here, we use N because the other polymer chains will be
        #   included as part of the prediction.
        f"and N <= {Config3D.MAX_LENGTH}"
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
            n = len(fam_group) if rfam == "" else Pipeline.TOP_N
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

    def prepare_targets(df, target_type):
        """Restore published columns and label one target type."""
        df = df.copy()
        df.columns = df.columns.str.replace("no.", "#")
        target_cols = og_cols.copy()
        for bl_name in Pipeline.BASELINES.keys():
            new_col = f"{bl_name.upper()} Sequence Homolog"
            target_cols += [
                new_col,
                f"{new_col} Date",
                f"{new_col} %id",
            ]
        for bl_name in Pipeline.BASELINES.keys():
            new_col = f"{bl_name.upper()} TM Homolog"
            target_cols += [
                new_col,
                f"{new_col} Date",
                f"{new_col} Rfam",
                f"{new_col} Score",
            ]
        df = df[target_cols].copy()
        df.insert(0, "type", target_type)
        df["Resolution"] = df["Resolution"].replace(NMR_SENTINEL, "N/A")
        return df

    targets = pd.concat(
        [prepare_targets(mon_df, "monomer"), prepare_targets(mul_df, "multimer")],
        ignore_index=True,
    ).sort_values(by=["type", "PDB ID", "Auth. Chain ID"])
    debug_df(targets)
    targets.to_parquet(Config3D.TARGET_FILE, compression="zstd", index=False)
    print(f"Wrote {len(targets)} targets to {Config3D.TARGET_FILE}")


if __name__ == "__main__":
    main()
