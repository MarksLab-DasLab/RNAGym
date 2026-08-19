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
            folder = resolve_source(score_cols_dict, model_name)[0]
            model_path = os.path.join(model_predictions_folder, folder, csv_file)
            if not os.path.exists(model_path):
                logger.warning(f"Model file {csv_file} not found in {model_name}")
                continue

            model_df = pd.read_csv(model_path)
            model_mutation_col = get_mutation_column(model_df)
            score_col = resolve_source(score_cols_dict, model_name)[1]
            if score_col not in model_df.columns:
                raise KeyError(
                    f"{model_path} has no column {score_col!r}; it holds "
                    f"{list(model_df.columns)}"
                )
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


def resolve_source(score_cols_dict, model_name):
    """
    Return the (folder, column) a model's scores are read from.

    An entry is either a bare column name, meaning the predictions live in a
    folder named after the model, or a dict giving the folder and column
    explicitly. The explicit form is what lets several entries read different
    columns of the same prediction files, which is how the four masked-marginal
    fill strategies are exposed: one run writes all four columns into one folder.
    """
    try:
        spec = score_cols_dict[model_name]
    except KeyError:
        raise KeyError(f"No score column configured for model {model_name!r}")
    if isinstance(spec, str):
        return model_name, spec
    return spec["folder"], spec["column"]


def four_fill_entries(name, folder, column_stem):
    """
    Register a masked language model's four fill strategies.

    A single scoring run writes one file per assay holding all four columns, so
    the four entries share a folder and differ only in the column they read.
    The fill named is what the model sees at a variant's OTHER mutated positions
    while one position is masked: see fitness/baselines/masked_lm.
    """
    return {
        f"{name}_{strategy}": {"folder": folder, "column": f"{column_stem}_{strategy}"}
        for strategy in ("wt_fill", "mask_fill", "mut_fill", "match_fill")
    }


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
    "rnagenesis": "rnagenesis_score",
    # AIDO.RNA size series, for the scaling comparison. Every checkpoint writes the
    # same aido_rna_score column, so they differ only by prediction folder.
    "aido_rna_1m": "aido_rna_score",
    "aido_rna_25m": "aido_rna_score",
    "aido_rna_300m": "aido_rna_score",
    "aido_rna_650m": "aido_rna_score",
    "EVmutation": "prediction_epistatic",
}

# The four fill strategies, for every masked language model that runs them. Each
# model has one prediction folder holding all four columns. These are not in
# ALL_MODELS: pass them to --models explicitly, since one model appears four
# times and a default merge should not multiply the released leaderboard.
FOUR_FILL_MODELS = []
for _name, _folder, _stem in [
    ("rna_fm", "rna_fm_4fill", "RNA_FM_scores"),
    ("rinalmo", "rinalmo_4fill", "logit_scores"),
    ("rnagenesis", "rnagenesis_4fill", "rnagenesis_score"),
    ("aido_rna", "aido_rna_4fill", "aido_rna_score"),
    ("aido_rna_1m", "aido_rna_1m_4fill", "aido_rna_score"),
    ("aido_rna_25m", "aido_rna_25m_4fill", "aido_rna_score"),
    ("aido_rna_300m", "aido_rna_300m_4fill", "aido_rna_score"),
    ("aido_rna_650m", "aido_rna_650m_4fill", "aido_rna_score"),
]:
    _entries = four_fill_entries(_name, _folder, _stem)
    SCORE_COLS.update(_entries)
    FOUR_FILL_MODELS.extend(_entries)

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
    "rnagenesis",
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
        "--models",
        nargs="+",
        default=None,
        help="Model entries to merge (default: every entry in ALL_MODELS). Use "
        "this to merge a subset, such as one masked model's four fill "
        "strategies: rna_fm_wt_fill rna_fm_mask_fill rna_fm_mut_fill rna_fm_match_fill",
    )
    parser.add_argument(
        "--assays_with_MSAs_only",
        action="store_true",
        help="Focus on assays with MSAs only (i.e., EVmutation)",
    )
    args = parser.parse_args()

    if args.assays_with_MSAs_only:
        model_list = ["EVmutation"]
    else:
        model_list = args.models if args.models else ALL_MODELS
    unknown = [m for m in model_list if m not in SCORE_COLS]
    if unknown:
        parser.error(f"No score column configured for: {unknown}")

    combine_csv_data(
        args.processed_folder,
        args.model_predictions_folder,
        args.output_folder,
        model_list,
        SCORE_COLS,
    )


if __name__ == "__main__":
    main()
