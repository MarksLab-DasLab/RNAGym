#!/usr/bin/env python3
"""Merge processed DMS assay CSVs with model prediction scores."""

import os
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_mutation_column(df):
    """Find the mutation column name in a DataFrame."""
    for col in ["mutant", "mutation", "Mutation", "mutations"]:
        if col in df.columns:
            return col
    raise ValueError("Couldn't find a mutation column in the dataframe")


def standardize_mutation(mutation):
    """Replace T with U and strip spaces (DNA -> RNA notation)."""
    return str(mutation).replace("T", "U").replace(" ", "")


def combine_csv_data(
    processed_folder,
    model_predictions_folder,
    output_folder,
    model_list,
    score_cols_dict,
):
    """Merge each assay CSV with model prediction scores via inner join on mutations."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for csv_file in os.listdir(processed_folder):
        if not csv_file.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(processed_folder, csv_file))
        mutation_col = get_mutation_column(df)
        df = df.dropna(subset=[mutation_col])
        df[mutation_col] = df[mutation_col].apply(standardize_mutation)

        save = True
        for model_name in model_list:
            model_path = os.path.join(model_predictions_folder, model_name, csv_file)
            if not os.path.exists(model_path):
                logger.warning(f"Model file {csv_file} not found in {model_name}")
                continue

            model_df = pd.read_csv(model_path)
            model_mutation_col = get_mutation_column(model_df)
            score_col = score_cols_dict[model_name]
            model_df = model_df[[model_mutation_col, score_col]]
            model_df.columns = [mutation_col, f"{model_name}_score"]
            model_df[mutation_col] = model_df[mutation_col].apply(standardize_mutation)
            model_df = model_df.dropna(subset=[mutation_col])
            model_df = model_df.drop_duplicates(subset=[mutation_col], keep="first")

            original_row_count = len(df)
            df = df.merge(model_df, on=mutation_col, how="inner")
            if len(df) != original_row_count:
                print(
                    f"Row count mismatch in {csv_file} after merging {model_name}: "
                    f"Expected {original_row_count}, but got {len(df)}"
                )
                save = False

        if len(df) > 0 and save:
            output_file = os.path.join(output_folder, csv_file)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved combined data to {output_file}")


SCORE_COLS = {
    "evo1": "evo_1_131k_base_score",
    "evo1.5": "evo_1.5_8k_base_score",
    "evo2": "evo2_7b_score",
    "evo2_40b": "evo2_40b_score",
    "GenSLM": "logit_scores",
    "NT": "kmer_pseudo_LL",
    "RNA-FM": "RNA_FM_scores",
    "rinalmo": "logit_scores",
    "RNAErnie": "Mutation_Scores",
    "orthrus": "orthrus_score",
    "aido_rna": "aido_rna_score",
    # AIDO.RNA size series, for the scaling comparison. Every checkpoint writes the
    # same aido_rna_score column, so they differ only by prediction folder.
    "aido_rna_1m": "aido_rna_score",
    "aido_rna_25m": "aido_rna_score",
    "aido_rna_300m": "aido_rna_score",
    "aido_rna_650m": "aido_rna_score",
    "EVmutation": "prediction_epistatic",
}

ALL_MODELS = [
    "evo1",
    "evo1.5",
    "evo2",
    "evo2_40b",
    "GenSLM",
    "NT",
    "rinalmo",
    "RNAErnie",
    "RNA-FM",
    "orthrus",
    "aido_rna",
    "aido_rna_1m",
    "aido_rna_25m",
    "aido_rna_300m",
    "aido_rna_650m",
    "EVmutation",
]


def main():
    parser = argparse.ArgumentParser(
        description="Combine processed CSV data with model predictions."
    )
    parser.add_argument(
        "--processed_folder",
        required=True,
        help="Path to the folder containing processed CSV files.",
    )
    parser.add_argument(
        "--model_predictions_folder",
        required=True,
        help="Path to the folder containing model prediction subfolders.",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Path to the folder where combined CSV files will be saved.",
    )
    parser.add_argument(
        "--assays_with_MSAs_only",
        action="store_true",
        help="Focus on assays with MSAs only (i.e., EVmutation)",
    )
    args = parser.parse_args()

    model_list = ["EVmutation"] if args.assays_with_MSAs_only else ALL_MODELS

    combine_csv_data(
        args.processed_folder,
        args.model_predictions_folder,
        args.output_folder,
        model_list,
        SCORE_COLS,
    )


if __name__ == "__main__":
    main()
