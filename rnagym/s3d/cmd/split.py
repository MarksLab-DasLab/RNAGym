"""Select the post-cutoff 3D benchmark targets."""

import pandas as pd
import polars as pl

from rnagym.config import Config2D, Config3D
from rnagym.s3d.util import Config as Pipeline
from rnagym.sequences import fitness_sequences, update_registry


def debug_df(df: pd.DataFrame) -> None:
    """Print a concise target summary."""
    df["Rfam E-value"] = pd.to_numeric(df["Rfam E-value"], errors="coerce")
    # Count Rfam hits (E-value < 1)
    rfam_hits = df.query("`Rfam E-value` < 1")["Rfam"].nunique()
    print(f"{rfam_hits} unique Rfam hits (E-value < 1)")

    # Count unique Rfam signatures (any detected Rfam)
    count = len(df.query('Rfam == "" or `Rfam E-value` > 10'))
    print(f"RNAs with Rfam E-value >10.0: {count}")

    # Average length
    avg_length = df["L"].mean()
    print(f"Average length: {avg_length:.0f} nt")

    # Average resolution among structures measured by diffraction or cryo-EM
    avg_res = df["Resolution"].mean()
    print(f"Average resolution: {avg_res:.2f} Å")


def get_split_candidates() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Select candidate monomer and multimer chains."""
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

    # Use the worst comma-separated resolution while leaving NMR null
    df["Resolution"] = (
        df["Resolution"]
        .str.split(",")
        .explode()
        .pipe(pd.to_numeric, errors="coerce")
        .groupby(level=0)
        .max()
    )

    df = df.loc[quality].copy()
    print(f"Quality RNAs: {len(df)}")

    df = df[df["Published"] > Config3D.TARGET_CUTOFF]
    print(f"After cutoff RNAs: {len(df)}")

    df.loc[:, "Rfam"] = df["Rfam"].fillna("")
    df = df[df["L"].between(Config3D.MIN_LENGTH, Config3D.MAX_SEQUENCE_LENGTH)]
    mon_df = df[monomers.loc[df.index]]
    print(f"Monomers: {len(mon_df)}")
    # Bound the annotated asymmetric-unit polymer length
    is_multimer = (df["% covered (any polymer)"] > Config3D.MAX_POLYMER_COVERAGE) & (
        df["N"] <= Config3D.MAX_COMPLEX_LENGTH
    )
    mul_df = df[is_multimer]
    print(f"Multimers: {len(mul_df)}")

    # Keep every experimental structure, including repeated exact sequences
    mon_df = mon_df.sort_values(
        by=["Resolution", "L", "PDB ID", "Asym. Chain ID"],
        ascending=[True, False, True, True],
    )
    mul_df = mul_df.sort_values(
        by=["Resolution", "L", "PDB ID", "Asym. Chain ID"],
        ascending=[True, False, True, True],
    )

    return mon_df, mul_df, og_cols


def main() -> None:
    """Write the selected targets and updated shared sequence registry."""
    mon_df, mul_df, og_cols = get_split_candidates()

    def prepare_targets(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
        """Restore published columns and label one target type."""
        df = df.copy()
        target_cols = og_cols.copy()
        for bl_name in Pipeline.BASELINES.keys():
            prefix = bl_name.upper()
            target_cols.extend(
                [
                    f"{prefix} TM Homolog",
                    f"{prefix} TM Homolog Date",
                    f"{prefix} TM Homolog Rfam",
                    f"{prefix} TM Homolog Score",
                ]
            )
        df = df.reindex(columns=target_cols)
        df.insert(0, "type", target_type)
        df["Resolution"] = pd.to_numeric(df["Resolution"])
        integer_columns = [
            "Rfam Cluster",
            "N_nt",
            "N_aa",
            "N",
            "L",
            "Rfam L",
            *[column for column in df if column.startswith("# of ")],
        ]
        df[integer_columns] = df[integer_columns].apply(pd.to_numeric).astype("Int64")
        return df

    targets = pd.concat(
        [prepare_targets(mon_df, "monomer"), prepare_targets(mul_df, "multimer")],
        ignore_index=True,
    ).sort_values(by=["type", "PDB ID", "Auth. Chain ID"])

    def update_homology(data: pd.DataFrame, cached: pd.DataFrame) -> pd.DataFrame:
        """Merge cached homology columns by PDB chain."""
        index = ["PDB ID", "Asym. Chain ID"]
        data = data.set_index(index)
        cached = cached.set_index(index)
        if data.index.has_duplicates or cached.index.has_duplicates:
            raise RuntimeError("PDB chain identifiers must be unique")
        for column in cached:
            if " TM Homolog" in column:
                values = cached[column].reindex(data.index)
                data[column] = values.combine_first(data[column])
        return data.reset_index()

    # Preserve cached training-homology annotations for previously released targets
    if Config3D.TARGET_FILE.is_file():
        targets = update_homology(targets, pd.read_parquet(Config3D.TARGET_FILE))
    if Config3D.USALIGN_ANNOTATION_DIR.is_dir():
        annotations = sorted(Config3D.USALIGN_ANNOTATION_DIR.glob("*.parquet"))
        if annotations:
            targets = update_homology(
                targets, pd.concat(map(pd.read_parquet, annotations))
            )

    structures = pl.read_parquet(Config2D.STRUCTURE_FILE).select("uid", "sequence")
    modalities = pl.concat(
        [
            pl.read_parquet(Config2D.MAPPING_FILE)
            .select("sequence")
            .with_columns(pl.lit("mapping").alias("modality")),
            structures.select(
                "sequence",
                pl.col("uid").str.split(":").list.first().alias("modality"),
            ),
            pl.from_pandas(targets[["Sequence (unmod.)"]])
            .rename({"Sequence (unmod.)": "sequence"})
            .with_columns(pl.lit("3d").alias("modality")),
            fitness_sequences(),
        ]
    )
    registry = update_registry(modalities)
    identifiers = pd.DataFrame(
        registry.select(
            pl.col("sequence").alias("Sequence (unmod.)"),
            "sequence_id",
            "cluster_rep",
        ).to_dict(as_series=False)
    )
    targets = targets.merge(identifiers, on="Sequence (unmod.)", validate="many_to_one")
    numeric_homology = [
        column for column in targets if column.endswith(" Homolog Score")
    ]
    targets[numeric_homology] = targets[numeric_homology].apply(pd.to_numeric)

    debug_df(targets)
    Config3D.SEQUENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    registry_tmp = Config3D.SEQUENCE_FILE.with_suffix(".parquet.tmp")
    targets_tmp = Config3D.TARGET_FILE.with_suffix(".parquet.tmp")
    registry.write_parquet(registry_tmp, compression="zstd", statistics=True)
    targets.to_parquet(targets_tmp, compression="zstd", index=False)
    registry_tmp.replace(Config3D.SEQUENCE_FILE)
    targets_tmp.replace(Config3D.TARGET_FILE)
    print(f"Wrote {registry.height:,} sequences to {Config3D.SEQUENCE_FILE}")
    print(f"Wrote {len(targets)} targets to {Config3D.TARGET_FILE}")


if __name__ == "__main__":
    main()
