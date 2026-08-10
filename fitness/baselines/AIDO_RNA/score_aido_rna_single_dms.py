#!/usr/bin/env python3
"""
Script to run AIDO.RNA inference on DMS assay sequences.
Takes a reference sheet and row ID to process specific assays.

AIDO.RNA (https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) is an encoder-only
transformer pretrained with a masked language modelling objective on 42M
non-coding RNA sequences from RNAcentral. We score variants with the
masked-marginal log-likelihood ratio, the same zero-shot proxy used by the other
masked RNA language-model baselines (RiNALMo, Orthrus):

    score(variant) = sum_i [ log P(mut_i | variant context, pos_i masked)
                             - log P(wt_i  | variant context, pos_i masked) ]

summed over the variant's mutated positions ``i``. Each position is masked in
the variant's own sequence, so for a multi-mutant the remaining mutations stay
in the context.

Masking one position of one variant is a single forward pass, but two variants
that differ only at the masked position share the same masked context. Those
contexts are deduplicated before inference, which is exact and cuts the number
of forward passes by up to 3x on single-substitution libraries.

The model code is the official implementation released by GenBio AI in the
``modelgenerator`` package (``pip install --no-deps modelgenerator``); only
torch and transformers are needed on top of it.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from modelgenerator.huggingface_models.rnabert import (
    RNABertForMaskedLM,
    RNABertTokenizer,
)

# AIDO.RNA accepts both alphabets and mirrors whichever one the context uses: on
# T-form input it puts ~0 probability on U and vice versa. Masked negative
# log-likelihood on the wild-type ncRNA constructs is consistently lower in
# T-form (e.g. 0.2197 vs 0.2223 on Domingo_2018_tRNA), so U is folded to T.
BASES = "ACGT"
MASK_CHAR = "#"


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess an RNA/DNA sequence for AIDO.RNA:
    - Convert to uppercase
    - Convert RNA (U) to DNA (T)
    - Remove any whitespace
    """
    return sequence.strip().upper().replace("U", "T")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AIDO.RNA (masked LM) inference on DMS assay sequences."
    )
    parser.add_argument(
        "--row_id",
        type=int,
        required=True,
        help="Row ID in the reference sheet to process",
    )
    parser.add_argument(
        "--ref_sheet",
        type=str,
        required=True,
        help="Path to reference sheet containing DMS_ID column",
    )
    parser.add_argument(
        "--dms_dir_path",
        type=str,
        required=True,
        help="Directory containing DMS CSV files",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="genbio-ai/AIDO.RNA-1.6B",
        help="AIDO.RNA model to use (default: genbio-ai/AIDO.RNA-1.6B)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Weight/compute dtype. bfloat16 is ~2x faster and was used for the "
        "released scores; float32 gives finer logits and fewer tied variants "
        "(default: bfloat16)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Maximum masked contexts scored per forward pass (default: 512)",
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=49152,
        help="Cap on batch_size x sequence length per forward pass, so that long "
        "assays automatically use a smaller batch (default: 49152, which peaks "
        "around 10 GB of GPU memory for the 1.6B model)",
    )
    return parser.parse_args()


def load_reference_data(ref_sheet_path: str, row_id: int) -> str:
    """
    Load reference sheet and get DMS_ID for specified row.

    Raises:
        ValueError: If row_id is not found or DMS_ID is missing
    """
    try:
        ref_df = pd.read_csv(ref_sheet_path)
        if row_id >= len(ref_df):
            raise ValueError(
                f"Row ID {row_id} exceeds number of rows in reference sheet"
            )

        dms_id = ref_df.loc[row_id, "DMS_ID"]
        if pd.isna(dms_id):
            raise ValueError(f"DMS_ID is missing for row {row_id}")

        return str(dms_id)

    except FileNotFoundError:
        raise FileNotFoundError(f"Reference sheet not found: {ref_sheet_path}")
    except KeyError:
        raise KeyError("Reference sheet must contain 'DMS_ID' column")


def load_dms_data(dms_dir_path: str, dms_id: str) -> pd.DataFrame:
    """
    Load DMS data for specified DMS_ID.

    Raises:
        FileNotFoundError: If DMS file is not found
    """
    dms_file = Path(dms_dir_path) / f"{dms_id}.csv"
    if not dms_file.exists():
        raise FileNotFoundError(f"DMS file not found: {dms_file}")

    df = pd.read_csv(dms_file)
    required_cols = ["mutant", "DMS_score", "sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DMS file: {missing_cols}")

    return df


def parse_mutations(mutant_str: str) -> list:
    """
    Parse a mutation string such as ``"A4T,A5G"`` into a list of
    ``(pos0, wt_base, mut_base)`` tuples with 0-based positions, in the DNA
    alphabet used by the model.

    Raises:
        ValueError: for non-substitution edits (indels) or unknown bases, so
        the caller can score the affected variant as NaN.
    """
    mutations = []
    for token in str(mutant_str).replace(" ", "").split(","):
        if not token:
            continue
        wt_base = token[0].upper().replace("U", "T")
        mut_base = token[-1].upper().replace("U", "T")
        pos = int(token[1:-1]) - 1  # 1-based -> 0-based
        if wt_base not in BASES or mut_base not in BASES:
            raise ValueError(f"Unsupported mutation token: {token}")
        mutations.append((pos, wt_base, mut_base))
    return mutations


def build_masking_tasks(mutants: list, sequences: list) -> tuple:
    """
    Expand each variant into one masked-scoring task per mutated position, and
    deduplicate identical masked contexts.

    A masked context is the variant's own sequence with the scored position
    replaced by a mask placeholder. Two variants that differ only at that
    position produce the same context and therefore the same log-probabilities,
    so the forward pass is shared. The context also determines the masked
    position, so it does not need to be tracked separately.

    Returns:
        contexts: list of masked context strings, one per forward pass
        ctx_tasks: list parallel to ``contexts``; entry k holds the
            ``(row_idx, wt_base, mut_base)`` tuples scored from context k
        scores: per-variant score array, pre-filled with 0.0 for scorable
            variants and NaN for wild-type / unparseable / out-of-range rows
    """
    scores = np.full(len(sequences), np.nan, dtype=float)
    ctx_index = {}
    contexts = []
    ctx_tasks = []

    for i, (mutant_str, seq) in enumerate(zip(mutants, sequences)):
        if pd.isna(mutant_str):
            continue  # wild-type row: leave as NaN
        try:
            mutations = parse_mutations(mutant_str)
            if not mutations:
                continue
            row_tasks = []
            for pos, wt_base, mut_base in mutations:
                if pos < 0 or pos >= len(seq):
                    raise ValueError(f"Mutation position {pos + 1} outside sequence")
                if seq[pos] != mut_base:
                    raise ValueError(
                        f"Sequence has {seq[pos]} at position {pos + 1}, "
                        f"expected the mutant base {mut_base}"
                    )
                row_tasks.append((seq[:pos] + MASK_CHAR + seq[pos + 1 :], wt_base, mut_base))
        except (ValueError, IndexError) as err:
            print(f"Skipping variant {mutant_str}: {err}")
            continue  # unsupported edit: leave as NaN
        for context, wt_base, mut_base in row_tasks:
            k = ctx_index.get(context)
            if k is None:
                k = len(contexts)
                ctx_index[context] = k
                contexts.append(context)
                ctx_tasks.append([])
            ctx_tasks[k].append((i, wt_base, mut_base))
        scores[i] = 0.0  # scorable: accumulate per-position deltas below

    return contexts, ctx_tasks, scores


def run_inference(
    model,
    tokenizer,
    contexts: list,
    ctx_tasks: list,
    scores: np.ndarray,
    device: str,
    batch_size: int,
    max_batch_tokens: int,
) -> np.ndarray:
    """
    Score masked contexts with AIDO.RNA's masked LM head and accumulate the
    per-position log-likelihood ratios into each variant's score.
    """
    base_ids = {b: tokenizer.convert_tokens_to_ids(b) for b in BASES}
    if tokenizer.unk_token_id in base_ids.values():
        raise ValueError(f"Tokenizer does not cover the {BASES} alphabet: {base_ids}")
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    unk_id = tokenizer.unk_token_id  # any base outside ACGT, e.g. an N in a construct

    # Sequence length is constant within an assay, so a single batch size is
    # enough; still derive it from the length so long assays stay in memory.
    seq_len = max(len(c) for c in contexts)
    batch_size = max(1, min(batch_size, max_batch_tokens // (seq_len + 2)))
    print(f"Using batch size {batch_size} for sequence length {seq_len}")

    for start in tqdm(
        range(0, len(contexts), batch_size), desc="Scoring", unit="batch"
    ):
        batch = contexts[start : start + batch_size]
        max_len = max(len(c) for c in batch) + 2  # [CLS] ... [SEP]

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        mask_pos = []
        for b, context in enumerate(batch):
            ids = [cls_id]
            for ch in context:
                ids.append(mask_id if ch == MASK_CHAR else base_ids.get(ch, unk_id))
            ids.append(sep_id)
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[b, : len(ids)] = 1
            # Dedup is only exact if a context masks exactly one position, which is
            # what makes the context string identify the position unambiguously.
            if context.count(MASK_CHAR) != 1:
                raise ValueError(f"Context does not mask exactly one position: {context}")
            mask_pos.append(context.index(MASK_CHAR) + 1)  # +1 for [CLS]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        rows = torch.arange(len(batch), device=device)
        cols = torch.tensor(mask_pos, device=device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[rows, cols].float(), dim=-1)
        log_probs = log_probs.cpu().numpy()

        for b in range(len(batch)):
            row_log_probs = log_probs[b]
            for row_idx, wt_base, mut_base in ctx_tasks[start + b]:
                scores[row_idx] += (
                    row_log_probs[base_ids[mut_base]] - row_log_probs[base_ids[wt_base]]
                )

    return scores


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load DMS ID from reference sheet
        dms_id = load_reference_data(args.ref_sheet, args.row_id)
        print(f"Processing DMS ID: {dms_id}")

        # Load DMS data
        dms_df = load_dms_data(args.dms_dir_path, dms_id)

        # Preprocess sequences (uppercase, RNA -> DNA)
        print("Preprocessing sequences...")
        sequences = [preprocess_sequence(seq) for seq in dms_df["sequence"].tolist()]

        # Expand variants into deduplicated masked contexts
        contexts, ctx_tasks, scores = build_masking_tasks(
            dms_df["mutant"].tolist(), sequences
        )
        n_tasks = sum(len(t) for t in ctx_tasks)
        print(
            f"Prepared {n_tasks} masked positions across "
            f"{int(np.sum(~np.isnan(scores)))} scorable variants "
            f"(of {len(sequences)} total), deduplicated to {len(contexts)} "
            f"forward passes"
        )
        if not contexts:
            raise ValueError("No scorable variants found")

        # Initialize model. The tokenizer vocabulary ships with modelgenerator.
        print(f"Initializing AIDO.RNA model ({args.model_name})...")
        vocab_file = os.path.join(
            os.path.dirname(__import__("modelgenerator").__file__),
            "huggingface_models",
            "rnabert",
            "vocab.txt",
        )
        tokenizer = RNABertTokenizer(vocab_file, version="v2")
        model = RNABertForMaskedLM.from_pretrained(
            args.model_name, torch_dtype=getattr(torch, args.dtype)
        )
        model = model.to(args.device)
        model.eval()

        # Run masked-marginal inference
        print("Running inference...")
        sequence_scores = run_inference(
            model,
            tokenizer,
            contexts,
            ctx_tasks,
            scores,
            args.device,
            args.batch_size,
            args.max_batch_tokens,
        )

        # Add scores to DataFrame
        score_column = "aido_rna_score"
        dms_df[score_column] = sequence_scores

        # Calculate Spearman correlation (ignoring unscored variants)
        correlation, pvalue = spearmanr(
            dms_df["DMS_score"], dms_df[score_column], nan_policy="omit"
        )

        # Save results
        output_file = output_dir / f"{dms_id}.csv"
        dms_df.to_csv(output_file, index=False)
        print(f"Saved results to: {output_file}")

        # Print summary statistics
        print("\nSummary:")
        print(f"Number of sequences: {len(sequences)}")
        print(
            f"Spearman correlation with DMS scores: {correlation:.3f} (p-value: {pvalue:.2e})"
        )
        print(f"Output saved to: {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# python score_aido_rna_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path aido_rna_output --model_name genbio-ai/AIDO.RNA-1.6B
