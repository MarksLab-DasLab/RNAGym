#!/usr/bin/env python3

###############################################################################
# `split.py`: Selects the post-cutoff 3D benchmark targets.
###############################################################################

import pandas as pd
import polars as pl

from rnagym.config import Config2D, Config3D
from rnagym.s3d.util import Config as Pipeline
from rnagym.sequences import fitness_sequences, update_registry


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

    # Comma-separated measurements use their worst resolution; NMR remains null
    df["Resolution"] = (
        df["Resolution"]
        .str.split(",")
        .explode()
        .pipe(pd.to_numeric, errors="coerce")
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

    df.loc[:, "Rfam"] = df["Rfam"].fillna("")

    # Structured
    df = df.query(f"{Config3D.MIN_LENGTH} <= L")
    mon_df = df[monomers.loc[df.index]]
    print(f"Monomers: {len(mon_df)}")
    is_multimer = (
        f"`% covered (any polymer)` > {Config3D.MAX_POLYMER_COVERAGE} "
        # NOTE(MCA): Here, we use N because the other polymer chains will be
        #   included as part of the prediction.
        f"and N <= {Config3D.MAX_COMPLEX_LENGTH}"
    )
    mul_df = df.query(is_multimer)
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


def main():
    mon_df, mul_df, og_cols = get_split_candidates()

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
        for column in target_cols:
            if column not in df:
                df[column] = None
        df = df[target_cols].copy()
        df.insert(0, "type", target_type)
        df["Resolution"] = pd.to_numeric(df["Resolution"])
        return df

    targets = pd.concat(
        [prepare_targets(mon_df, "monomer"), prepare_targets(mul_df, "multimer")],
        ignore_index=True,
    ).sort_values(by=["type", "PDB ID", "Auth. Chain ID"])

    def update_homology(data, cached):
        """Merge cached homology columns by PDB chain."""
        index = ["PDB ID", "Asym. Chain ID"]
        data = data.set_index(index)
        cached = cached.set_index(index)
        data.update(cached[[column for column in cached if "Homolog" in column]])
        return data.reset_index()

    # Preserve cached training-homology annotations for previously released targets
    if Config3D.TARGET_FILE.is_file():
        targets = update_homology(targets, pd.read_parquet(Config3D.TARGET_FILE))
    if Config3D.USALIGN_ANNOTATION_DIR.is_dir():
        annotations = list(Config3D.USALIGN_ANNOTATION_DIR.glob("*.parquet"))
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
        column
        for column in targets
        if column.endswith(" %id") or column.endswith(" Homolog Score")
    ]
    targets[numeric_homology] = targets[numeric_homology].apply(pd.to_numeric)

    debug_df(targets)
    Config2D.SEQUENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    registry_tmp = Config2D.SEQUENCE_FILE.with_suffix(".parquet.tmp")
    targets_tmp = Config3D.TARGET_FILE.with_suffix(".parquet.tmp")
    registry.write_parquet(registry_tmp, compression="zstd", statistics=True)
    targets.to_parquet(targets_tmp, compression="zstd", index=False)
    registry_tmp.replace(Config2D.SEQUENCE_FILE)
    targets_tmp.replace(Config3D.TARGET_FILE)
    print(f"Wrote {registry.height:,} sequences to {Config2D.SEQUENCE_FILE}")
    print(f"Wrote {len(targets)} targets to {Config3D.TARGET_FILE}")


if __name__ == "__main__":
    main()
