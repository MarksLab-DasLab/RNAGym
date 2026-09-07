"""Score Evo 1 and 1.5 in float32 with a BOS token and mean log probability."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from rnagym.config import ConfigFitness
from rnagym.fitness.data import (
    read_assay,
    read_reference,
    write_csv_atomically,
)


def main() -> None:
    """Run the Evo scoring command."""
    run(parse_args())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the checkpoint, assay and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", default="all", help="Reference rows: all (default), 12 or 0-8,12"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", dest="model_name", default="evo-1.5-8k-base")
    return parser.parse_args(argv)


def prepare_model(model: Any) -> Any:
    """Use float32 inference to preserve small score differences across GPUs."""
    from flash_attn.modules.mha import CrossAttention, SelfAttention

    # BF16 rounding changes variant rankings, and FP16 overflows for both checkpoints
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.float()
    for block in model.blocks:
        if hasattr(block, "inner_mha_cls"):
            attention = block.inner_mha_cls
            attention.use_flash_attn = False
            attention.inner_attn = SelfAttention(causal=True)
            attention.inner_cross_attn = CrossAttention(causal=True)
    return model


def run(
    args: argparse.Namespace, model_factory: Callable[..., Any] | None = None
) -> None:
    """Load one checkpoint and score the selected assays."""
    reference = read_reference(ConfigFitness.REFERENCE_FILE, args.rows)
    checkpoint = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for name in reference["DMS_ID"]:
        assay = read_assay(ConfigFitness.ASSAY_DIR / f"{name}.csv")
        raw = assay["sequence"]
        if (
            assay.is_empty()
            or raw.is_null().any()
            or raw.str.strip_chars().eq("").any()
        ):
            raise ValueError(f"{name} has missing or empty sequences")
        sequences = (
            raw.str.strip_chars().str.to_uppercase().str.replace_all("U", "T").to_list()
        )
        if len({len(sequence) for sequence in sequences}) != 1:
            raise ValueError(f"{name} has mixed sequence lengths")
        if checkpoint is None:
            if model_factory is None:
                from rnagym.fitness.baselines.Evo.checkpoints import load_evo

                model_factory = load_evo
            checkpoint = model_factory(args.model_name)
            prepare_model(checkpoint.model.to(device)).eval()
        model = checkpoint.model
        scores = run_inference(
            model, checkpoint.tokenizer, sequences, device, 128
        ).mean(axis=1)
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"{name} produced nonfinite model scores")
        column = f"{args.model_name.replace('-', '_')}_score"
        result = assay.with_columns(pl.Series(column, scores))
        write_csv_atomically(result, args.output / f"{name}.csv")
        print(
            f"{name}: Spearman {spearmanr(assay['DMS_score'].to_numpy(), scores).statistic:.6f}"
        )


def run_inference(
    model: Any, tokenizer: Any, sequences: list[str], device: str, batch_size: int
) -> np.ndarray:
    """Return float32 nucleotide log probabilities in input order, excluding BOS."""
    from evo.scoring import logits_to_logprobs, prepare_batch

    # Bound unfused attention memory for the longer coding assays
    batch_size = min(batch_size, max(1, 4096 // (len(sequences[0]) + 1)))
    batches = []
    for start in tqdm(range(0, len(sequences), batch_size), desc="Scoring"):
        inputs, _ = prepare_batch(
            sequences[start : start + batch_size],
            tokenizer,
            prepend_bos=True,
            device=device,
        )
        with torch.inference_mode():
            logits, *_ = model(inputs)
            values = logits_to_logprobs(logits, inputs, trim_bos=True)
        batches.append(values.float().cpu().numpy())
    return np.concatenate(batches)


if __name__ == "__main__":
    main()
