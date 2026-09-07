"""Score GenSLM with mean next-codon log likelihood, excluding padding."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rnagym.config import ConfigFitness
from rnagym.fitness.data import (
    read_assay,
    read_reference,
    write_csv_atomically,
)


def main(
    args: argparse.Namespace | None = None,
    model_factory: Callable[..., Any] | None = None,
) -> None:
    """Load GenSLM once and score the selected assays."""
    args = args or parse_args()
    reference = read_reference(ConfigFitness.REFERENCE_FILE, args.rows)
    if model_factory is None:
        from genslm import GenSLM

        model_factory = GenSLM
    model = model_factory(
        "genslm_2.5B_patric",
        model_cache_dir=str(ConfigFitness.CHECKPOINT_DIR / "genslm/2.5B"),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for row in reference.iter_rows(named=True):
        process_single_row(
            row,
            model,
            device,
            ConfigFitness.ASSAY_DIR,
            args.output,
            "logit_scores",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the assays and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", default="all", help="Reference rows: all (default), 12 or 0-8,12"
    )
    parser.add_argument(
        "--output", type=Path, default=ConfigFitness.PREDICTION_DIR / "GenSLM"
    )
    return parser.parse_args(argv)


def process_single_row(
    row: dict[str, str],
    model: Any,
    device: str,
    base_dir: Path,
    results_dir: Path,
    score_column: str,
) -> None:
    """Score an assay with higher values indicating greater sequence likelihood."""
    name = row["DMS_ID"]
    assay = read_assay(base_dir / f"{name}.csv").drop_nulls("mutant")
    raw = assay["sequence"]
    if assay.is_empty() or raw.is_null().any() or raw.str.strip_chars().eq("").any():
        raise ValueError(f"{name} has missing or empty sequences")
    sequences = (
        raw.str.strip_chars().str.to_uppercase().str.replace_all("U", "T").to_list()
    )
    scores = sequence_log_likelihoods(model, sequences, device)
    result = assay.with_columns(
        pl.Series("mutated_sequence", sequences), pl.Series(score_column, scores)
    )
    write_csv_atomically(result, results_dir / f"{name}.csv")


def sequence_log_likelihoods(
    model: Any, sequences: list[str], device: str, batch_size: int = 4
) -> np.ndarray:
    """Return mean causal log likelihoods using the official codon tokenizer.

    Each logit predicts the following token. The first token has no prediction,
    and padding is excluded from both the loss and its denominator. This equals
    the negative native model loss for an unpadded sequence.
    """
    from genslm import SequenceDataset

    if batch_size < 1 or not sequences:
        raise ValueError("Scoring requires sequences and a positive batch size")
    if any(not sequence or set(sequence) - set("ACGT") for sequence in sequences):
        raise ValueError("GenSLM sequences must contain only A, C, G and T")
    max_length = max(map(len, sequences))
    token_count = (max_length + 2) // 3 + model.tokenizer.num_special_tokens_to_add()
    if token_count > model.seq_length:
        raise ValueError("Sequence exceeds the published GenSLM input length limit")
    if token_count < 2:
        raise ValueError("GenSLM requires at least two tokens per sequence")
    dataset = SequenceDataset(sequences, token_count, model.tokenizer)
    batches = []
    with torch.inference_mode():
        for batch in tqdm(DataLoader(dataset, batch_size=batch_size), desc="Scoring"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].squeeze(1).to(device)
            targets = input_ids[:, 1:].masked_fill(~attention_mask[:, 1:].bool(), -100)
            counts = attention_mask[:, 1:].sum(1)
            if (counts == 0).any():
                raise ValueError("GenSLM requires at least two tokens per sequence")
            output = model(input_ids, attention_mask, output_hidden_states=False)
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].float().permute(0, 2, 1),
                targets,
                reduction="none",
            )
            batches.append((-loss.sum(1) / counts).cpu().numpy())
    scores = np.concatenate(batches)
    if not np.isfinite(scores).all():
        raise FloatingPointError("GenSLM produced nonfinite model scores")
    return scores


if __name__ == "__main__":
    main()
