#!/usr/bin/env python3
"""Merge processed DMS assay CSVs with model prediction scores."""

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd

if __package__:
    from .model_registry import (
        ALL_MODELS,
        ASSAY_GROUPS,
        SCORE_COLS,
        resolve_source,
    )
else:
    from model_registry import ALL_MODELS, ASSAY_GROUPS, SCORE_COLS, resolve_source

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
    allow_incomplete=False,
    assay_types=None,
    assay_group="all",
):
    """
    Merge each assay CSV with model prediction scores via inner join on mutations.

    Every requested model must cover every assay in the selected group.
    The coverage check runs before any output is written, and all merged files
    are staged before they replace prior outputs.
    """
    if len(model_list) != len(set(model_list)):
        raise ValueError(f"Model list contains duplicates: {model_list}")
    processed_files = sorted(Path(processed_folder).glob("*.csv"))
    if not processed_files:
        raise FileNotFoundError(f"No assay CSVs found under {processed_folder}")
    if assay_group != "all" and assay_group not in ASSAY_GROUPS:
        raise ValueError(f"Unknown assay group: {assay_group!r}")
    if assay_types is not None:
        unknown = sorted(
            path.stem for path in processed_files if path.stem not in assay_types
        )
        if unknown:
            raise ValueError(
                f"Processed assays are absent from the reference sheet: {unknown[:5]}"
            )
    if assay_group != "all":
        if assay_types is None:
            raise ValueError("assay_types is required when selecting an assay group")
        selected_types = ASSAY_GROUPS[assay_group]
        processed_files = [
            path for path in processed_files if assay_types[path.stem] in selected_types
        ]
        if not processed_files:
            raise FileNotFoundError(
                f"No {assay_group} assay CSVs found under {processed_folder}"
            )
    processed_names = {path.name for path in processed_files}

    expected = {}
    for model_name in model_list:
        if model_name == "EVmutation":
            expected[model_name] = set()
        else:
            expected[model_name] = processed_names

    missing = {}
    for model_name, assay_names in expected.items():
        folder, _ = resolve_source(score_cols_dict, model_name)
        absent = sorted(
            name
            for name in assay_names
            if not (Path(model_predictions_folder) / folder / name).is_file()
        )
        if absent:
            missing[model_name] = absent
    if missing and not allow_incomplete:
        detail = " | ".join(
            f"{model}: {len(files)} missing, including {files[:3]}"
            for model, files in sorted(missing.items())
        )
        raise FileNotFoundError(
            f"Prediction coverage is incomplete under {model_predictions_folder}: {detail}"
        )

    output_path = Path(output_folder)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contributed = {model: set() for model in model_list}

    with tempfile.TemporaryDirectory(
        dir=output_path.parent, prefix=f".{output_path.name}-"
    ) as staging_dir:
        staging_path = Path(staging_dir)
        for processed_file in processed_files:
            csv_file = processed_file.name

            df = pd.read_csv(processed_file)
            mutation_col = get_mutation_column(df)
            df = df.dropna(subset=[mutation_col])
            df[mutation_col] = df[mutation_col].apply(standardize_mutation)

            for model_name in model_list:
                folder, score_col = resolve_source(score_cols_dict, model_name)
                model_path = Path(model_predictions_folder) / folder / csv_file
                if not model_path.exists():
                    logger.warning(f"Model file {csv_file} not found in {model_name}")
                    continue

                model_df = pd.read_csv(model_path)
                model_mutation_col = get_mutation_column(model_df)
                if score_col not in model_df.columns:
                    raise KeyError(
                        f"{model_path} has no column {score_col!r}. It holds "
                        f"{list(model_df.columns)}"
                    )
                model_df = model_df[[model_mutation_col, score_col]]
                model_df.columns = [mutation_col, f"{model_name}_score"]
                model_df = model_df.dropna(subset=[mutation_col])
                model_df[mutation_col] = model_df[mutation_col].apply(
                    standardize_mutation
                )
                duplicated = model_df[model_df.duplicated(mutation_col, keep=False)]
                if not duplicated.empty:
                    conflicting = duplicated.groupby(mutation_col, dropna=False)[
                        f"{model_name}_score"
                    ].nunique(dropna=False)
                    conflicting = conflicting[conflicting > 1]
                    if not conflicting.empty:
                        raise ValueError(
                            f"{model_path} has conflicting scores for duplicate "
                            f"mutations: {list(conflicting.index[:5])}"
                        )
                model_df = model_df.drop_duplicates(subset=[mutation_col], keep="first")

                original_row_count = len(df)
                df = df.merge(model_df, on=mutation_col, how="inner")
                if len(df) != original_row_count:
                    raise ValueError(
                        f"Row count mismatch in {csv_file} after merging {model_name}: "
                        f"Expected {original_row_count}, but got {len(df)}"
                    )
                contributed[model_name].add(csv_file)

            df.to_csv(staging_path / csv_file, index=False)

        absent = sorted(model for model, assays in contributed.items() if not assays)
        if absent and not allow_incomplete:
            raise FileNotFoundError(
                f"No predictions found for {absent} under {model_predictions_folder}. "
                "Check the folder names against SCORE_COLS, or pass --allow_incomplete "
                "to merge without them."
            )

        output_path.mkdir(parents=True, exist_ok=True)
        staged_names = {path.name for path in staging_path.iterdir()}
        for stale_file in output_path.glob("*.csv"):
            if stale_file.name not in staged_names:
                stale_file.unlink()
        for staged_file in staging_path.iterdir():
            staged_file.replace(output_path / staged_file.name)

    for model_name, assays in sorted(contributed.items()):
        logger.info(f"{model_name}: {len(assays)} assays merged")


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
        "--reference_file",
        type=str,
        default=str(Path(__file__).with_name("reference_sheet_final.csv")),
        help="Reference sheet used to identify the assays",
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
        "--type",
        default="ncRNA",
        choices=["all", *ASSAY_GROUPS],
        help="Assay group to merge (default: ncRNA)",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Merge even when a requested model has no predictions at all. Off by "
        "default, so that a misspelled prediction folder fails instead of "
        "producing a merge with that model silently missing",
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

    reference = pd.read_csv(args.reference_file, encoding="utf-8-sig")
    required = {"DMS_ID", "RNA_TYPE"}
    missing = sorted(required - set(reference.columns))
    if missing:
        parser.error(f"Reference sheet is missing columns: {missing}")
    if reference[list(required)].isna().any().any():
        parser.error("Reference sheet has missing DMS_ID or RNA_TYPE values")
    duplicated = sorted(
        reference.loc[reference["DMS_ID"].duplicated(), "DMS_ID"].astype(str)
    )
    if duplicated:
        parser.error(f"Reference sheet repeats DMS_ID values: {duplicated[:5]}")
    assay_types = dict(zip(reference["DMS_ID"], reference["RNA_TYPE"]))

    combine_csv_data(
        args.processed_folder,
        args.model_predictions_folder,
        args.output_folder,
        model_list,
        SCORE_COLS,
        allow_incomplete=args.allow_incomplete,
        assay_types=assay_types,
        assay_group=args.type,
    )


if __name__ == "__main__":
    main()
