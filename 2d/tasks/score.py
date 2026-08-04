#!/usr/bin/env python3

"""Score secondary structure model predictions."""

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from tqdm.auto import tqdm

from tasks.utils import load_registry

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data" / "2d" / "chemical_mapping"
MAPPING_FILE = DATA_DIR / "rnagym_mapping.parquet"
STRUCTURE_FILE = DATA_DIR / "rnagym_2d.parquet"
SEQUENCE_FILE = DATA_DIR / "rnagym_sequences.parquet"
PREDICTION_DIR = DATA_DIR / "predictions"
LEADERBOARD_FILE = REPO_DIR / "leaderboard" / "2d" / "leaderboard.csv"
BATCH_SIZE = 8192
TRAINING_OVERLAP = {"eternafold", "ribonanzanet"}

OPENERS = "([{<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CLOSERS = ")]}>abcdefghijklmnopqrstuvwxyz"
PAIRS = dict(zip(OPENERS, CLOSERS))


def prediction_files(model: str, dataset: str) -> list[Path]:
    """Return the prediction shards for one model and dataset."""
    files = sorted(
        (PREDICTION_DIR / model / dataset).glob("*.parquet"),
        key=lambda path: int(path.stem),
    )
    if not files:
        raise FileNotFoundError(f"No {dataset} predictions found for {model}")
    return files


def parse_pairs(structure: str) -> set[tuple[int, int]]:
    """Parse base pairs from extended dot-bracket notation."""
    stacks = {opener: [] for opener in OPENERS}
    openers_by_closer = {closer: opener for opener, closer in PAIRS.items()}
    pairs = set()
    for position, symbol in enumerate(structure):
        if symbol == ".":
            continue
        if symbol in stacks:
            stacks[symbol].append(position)
        elif symbol in openers_by_closer:
            opener = openers_by_closer[symbol]
            if not stacks[opener]:
                raise ValueError(f"Unmatched {symbol} in {structure}")
            pairs.add((stacks[opener].pop(), position))
        else:
            raise ValueError(f"Unknown structure symbol {symbol}")
    if any(stacks.values()):
        raise ValueError(f"Unmatched opener in {structure}")
    return pairs


def structure_f1(reference: str, prediction: str) -> float:
    """Calculate base-pair F1 between two dot-bracket structures."""
    if len(reference) != len(prediction):
        raise ValueError("Reference and predicted structures have different lengths")
    reference_pairs = parse_pairs(reference)
    predicted_pairs = parse_pairs(prediction)
    total_pairs = len(reference_pairs) + len(predicted_pairs)
    if not total_pairs:
        return 1.0
    return 2 * len(reference_pairs & predicted_pairs) / total_pairs


def load_mapping() -> pl.DataFrame:
    """Load chemical mapping profiles."""
    return pl.read_parquet(
        MAPPING_FILE,
        columns=["uid", "sequence_id", "sequence", "modifier", "reactivity"],
    )


def load_assignments() -> pl.DataFrame:
    """Load sequence cluster and fold assignments."""
    return load_registry(SEQUENCE_FILE).select("sequence_id", "cluster_rep", "fold")


def spearman_rows(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Calculate Spearman correlations across rows."""
    valid = np.isfinite(reference) & np.isfinite(prediction)
    reference = rankdata(np.where(valid, reference, np.nan), axis=1, nan_policy="omit")
    prediction = rankdata(
        np.where(valid, prediction, np.nan), axis=1, nan_policy="omit"
    )
    means = (valid.sum(axis=1) + 1) / 2
    reference -= means[:, None]
    prediction -= means[:, None]
    numerator = np.nansum(reference * prediction, axis=1)
    denominator = np.sqrt(
        np.nansum(reference**2, axis=1) * np.nansum(prediction**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(reference.shape[0]),
        where=denominator > 0,
    )


def score_mapping_batch(batch: pl.DataFrame, modifier: str, length: int) -> np.ndarray:
    """Score one rectangular batch of chemical mapping profiles."""
    reactivity = batch["reactivity"].list.to_array(length).to_numpy()
    probabilities = batch["probabilities"].list.to_array(length).to_numpy()
    if modifier in {"DMS", "CMCT"}:
        bases = np.frombuffer(
            "".join(batch["sequence"].to_list()).encode(), dtype=np.uint8
        ).reshape(-1, length)
        if modifier == "DMS":
            measured = (bases == ord("A")) | (bases == ord("C"))
        else:
            measured = (bases == ord("G")) | (bases == ord("U"))
        reactivity = np.where(measured, reactivity, np.nan)
    return spearman_rows(reactivity, 1 - probabilities)


def score_mapping(
    model: str, profiles: pl.DataFrame, registry: pl.DataFrame
) -> pl.DataFrame:
    """Score one model against all chemical mapping profiles."""
    expected_profiles = profiles.height
    predictions = pl.read_parquet(
        prediction_files(model, "mapping"),
        columns=["sequence_id", "probabilities"],
    )
    if predictions["sequence_id"].n_unique() != predictions.height:
        raise RuntimeError(f"Duplicate chemical mapping predictions for {model}")
    profiles = (
        profiles.join(predictions, on="sequence_id")
        .join(registry, on="sequence_id")
        .with_columns(pl.col("sequence").str.len_chars().alias("length"))
    )
    if (
        profiles.height != expected_profiles
        or profiles["uid"].n_unique() != expected_profiles
    ):
        raise RuntimeError(
            f"Incomplete or duplicate chemical mapping predictions for {model}"
        )
    if profiles.filter(pl.col("length") != pl.col("probabilities").list.len()).height:
        raise ValueError(f"Prediction length mismatch for {model}")

    scores = []
    with tqdm(
        total=profiles.height,
        desc=f"Scoring {model} chemical mapping",
        unit="profiles",
    ) as progress:
        for (modifier, length), group in profiles.group_by("modifier", "length"):
            for batch in group.iter_slices(BATCH_SIZE):
                scores.append(
                    batch.select(
                        "uid", "modifier", "sequence_id", "cluster_rep", "fold"
                    ).with_columns(
                        pl.Series("score", score_mapping_batch(batch, modifier, length))
                    )
                )
                progress.update(batch.height)
    return (
        pl.concat(scores)
        .with_columns(
            pl.lit(model).alias("model"),
            pl.lit("mapping").alias("dataset"),
            pl.lit("unpaired_probability").alias("method"),
            pl.lit("spearman").alias("metric"),
            pl.col("modifier").alias("modality"),
        )
        .select(
            "model",
            "dataset",
            "modality",
            "method",
            "metric",
            "uid",
            "sequence_id",
            "cluster_rep",
            "fold",
            "score",
        )
        .sort("uid")
    )


def score_structures(model: str, registry: pl.DataFrame) -> pl.DataFrame:
    """Score one model against all discrete structures."""
    references = pl.read_parquet(STRUCTURE_FILE).join(registry, on="sequence_id")
    predictions = pl.read_parquet(
        prediction_files(model, "2d"),
        columns=["sequence_id", "structures"],
    )
    expected_sequences = references["sequence_id"].n_unique()
    if (
        predictions.height != expected_sequences
        or predictions["sequence_id"].n_unique() != expected_sequences
    ):
        raise RuntimeError(f"Incomplete or duplicate 2D predictions for {model}")

    scores = references.join(predictions, on="sequence_id")
    if scores.height != references.height:
        raise RuntimeError(f"Incomplete 2D predictions for {model}")

    return (
        scores.explode("structures", empty_as_null=True)
        .unnest("structures")
        .with_columns(
            pl.struct("secondary_structure", "dot_bracket")
            .map_elements(
                lambda row: structure_f1(
                    row["secondary_structure"], row["dot_bracket"]
                ),
                return_dtype=pl.Float64,
            )
            .alias("score"),
            pl.lit(model).alias("model"),
            pl.lit("2d").alias("dataset"),
            pl.col("uid").str.split(":").list.first().alias("modality"),
            pl.lit("f1").alias("metric"),
        )
        .select(
            "model",
            "dataset",
            "modality",
            "method",
            "metric",
            "uid",
            "sequence_id",
            "cluster_rep",
            "fold",
            "score",
        )
        .sort(["method", "uid"])
    )


def summarize(scores: pl.DataFrame) -> pl.DataFrame:
    """Macro-average scores across sequence clusters."""
    groups = ["model", "dataset", "modality", "method", "metric"]
    return (
        scores.group_by(*groups, "cluster_rep")
        .agg(pl.col("score").mean(), pl.len().alias("samples"))
        .group_by(groups)
        .agg(
            pl.col("score").mean(),
            pl.len().alias("clusters"),
            pl.col("samples").sum(),
        )
        .sort(["dataset", "score"], descending=[False, True])
    )


def score_model(
    model: str, profiles: pl.DataFrame, registry: pl.DataFrame
) -> pl.DataFrame:
    """Score one model against both benchmark datasets."""
    return pl.concat(
        [
            score_mapping(model, profiles, registry),
            score_structures(model, registry),
        ]
    )


def main() -> None:
    """Score every model and write the leaderboard."""
    models = sorted(path.name for path in PREDICTION_DIR.iterdir() if path.is_dir())
    if not models:
        raise FileNotFoundError(f"No model predictions found in {PREDICTION_DIR}")

    profiles = load_mapping()
    registry = load_assignments()
    summaries = []
    for model in models:
        scores = score_model(model, profiles, registry)
        summaries.append(summarize(scores))

    leaderboard = (
        pl.concat(summaries)
        .with_columns(
            (
                (pl.col("dataset") == "mapping")
                & pl.col("model").is_in(TRAINING_OVERLAP)
            ).alias("training_overlap"),
            pl.col("score")
            .rank("dense", descending=True)
            .over("dataset", "modality", "metric")
            .cast(pl.UInt16)
            .alias("rank"),
        )
        .select(
            "rank",
            "model",
            "dataset",
            "modality",
            "method",
            "metric",
            "score",
            "clusters",
            "samples",
            "training_overlap",
        )
        .sort("dataset", "modality", "rank", "model", "method")
    )
    leaderboard = leaderboard.with_columns(
        pl.when("training_overlap")
        .then(pl.col("model") + "*")
        .otherwise(pl.col("model"))
        .alias("model")
    )
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.write_csv(LEADERBOARD_FILE)

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(leaderboard.drop("training_overlap"))
    print(f"Wrote leaderboard to {LEADERBOARD_FILE}")
    print("* Model training data overlap the chemical mapping benchmark")


if __name__ == "__main__":
    main()
