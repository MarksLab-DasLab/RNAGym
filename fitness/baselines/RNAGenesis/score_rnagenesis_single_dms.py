#!/usr/bin/env python3
"""
Script to run RNAGenesis inference on DMS assay sequences.
Takes a reference sheet and row ID to process specific assays.

RNAGenesis (https://github.com/zaixizhang/RNAGenesis) is a generalist RNA
foundation model. Its released encoder (https://huggingface.co/Zaixi/RNAGenesis)
is an xTrimoPGLM-style bidirectional transformer pretrained with a masked
language-modelling objective on RNAcentral, exposed as
``xTrimoPGLMForMaskedLM``. We score variants with the masked-marginal
log-likelihood ratio, the same zero-shot proxy used by the other masked RNA
language-model baselines (RiNALMo, Orthrus, AIDO.RNA):

    score(variant) = sum_i [ log P(mut_i | variant context, pos_i masked)
                             - log P(wt_i  | variant context, pos_i masked) ]

summed over the variant's mutated positions ``i``. Each position is masked in
the variant's own sequence, so for a multi-mutant the remaining mutations stay
in the context.

Two properties of the released checkpoint drive the implementation:

1. The model uses the RNA alphabet. Its vocabulary contains U and no T, so
   sequences are folded to U (the opposite of AIDO.RNA).
2. The shipped tokenizer wrapper cannot be used directly. ``<mask>``, ``<unk>``
   and ``<eos>`` are absent from the vocabulary and receive phantom
   added-token ids past the real vocab, ``mask_token_id`` is None, and
   ``convert_tokens_to_ids`` iterates a string character by character. The
   author's own ``run.py`` depends on that character iteration and adds no
   special tokens, so input ids are built directly from ``tokenizer.model``:
   one id per nucleotide, no CLS/EOS.

The mask token is ``tMASK``, the token-level mask of the xTrimoPGLM family. It
was confirmed empirically: on wild-type ncRNA constructs it gives a lower masked
negative log-likelihood than gMASK or sMASK, and puts >99.9% of the predicted
mass on A/C/G/U.

Masking one position of one variant is a single forward pass, but two variants
that differ only at the masked position share the same masked context. Those
contexts are deduplicated before inference, which is exact and cuts the number
of forward passes by up to 3x on single-substitution libraries.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM

# RNAGenesis uses the RNA alphabet: its vocabulary has U and no T.
BASES = "ACGU"
MASK_TOKEN = "tMASK"
UNK_TOKEN = "N"
MASK_CHAR = "#"


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess an RNA/DNA sequence for RNAGenesis:
    - Convert to uppercase
    - Convert DNA (T) to RNA (U)
    - Remove any whitespace
    """
    return sequence.strip().upper().replace("T", "U")


def load_vocab(model_path: str) -> dict:
    """
    Read the model's own vocabulary file and return a token -> id mapping.

    The HuggingFace tokenizer wrapper is bypassed on purpose, see module
    docstring.
    """
    vocab_file = Path(model_path) / "tokenizer.model"
    if not vocab_file.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_file}. Point --model_name at a "
            f"local copy of the RNAGenesis checkpoint."
        )
    tokens = vocab_file.read_text().splitlines()
    return {token: index for index, token in enumerate(tokens)}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RNAGenesis (masked LM) inference on DMS assay sequences."
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
        required=True,
        help="Local directory holding the RNAGenesis checkpoint. The released "
        "repo omits quantization.py, which the modelling code imports, so a "
        "local copy with that file added is required.",
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
        default=256,
        help="Maximum masked contexts scored per forward pass (default: 256)",
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=32768,
        help="Cap on batch_size x sequence length per forward pass, so that long "
        "assays automatically use a smaller batch (default: 32768)",
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
    Parse a mutation string such as ``"A4U,A5G"`` into a list of
    ``(pos0, wt_base, mut_base)`` tuples with 0-based positions, in the RNA
    alphabet used by the model.

    Raises:
        ValueError: for non-substitution edits (indels) or unknown bases, so
        the caller can score the affected variant as NaN.
    """
    mutations = []
    for token in str(mutant_str).replace(" ", "").split(","):
        if not token:
            continue
        wt_base = token[0].upper().replace("T", "U")
        mut_base = token[-1].upper().replace("T", "U")
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
    vocab: dict,
    contexts: list,
    ctx_tasks: list,
    scores: np.ndarray,
    device: str,
    batch_size: int,
    max_batch_tokens: int,
) -> np.ndarray:
    """
    Score masked contexts with the RNAGenesis masked LM head and accumulate the
    per-position log-likelihood ratios into each variant's score.

    Input ids are one per nucleotide with no special tokens, matching the
    reference usage shipped with the checkpoint, so the masked position index in
    the token sequence equals the position in the RNA sequence.
    """
    base_ids = {b: vocab[b] for b in BASES}
    mask_id = vocab[MASK_TOKEN]
    pad_id = vocab["<pad>"]
    unk_id = vocab[UNK_TOKEN]  # any base outside ACGU, e.g. an N in a construct

    seq_len = max(len(c) for c in contexts)
    batch_size = max(1, min(batch_size, max_batch_tokens // seq_len))
    print(f"Using batch size {batch_size} for sequence length {seq_len}")

    for start in tqdm(
        range(0, len(contexts), batch_size), desc="Scoring", unit="batch"
    ):
        batch = contexts[start : start + batch_size]
        max_len = max(len(c) for c in batch)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        mask_pos = []
        for b, context in enumerate(batch):
            ids = [
                mask_id if ch == MASK_CHAR else base_ids.get(ch, unk_id)
                for ch in context
            ]
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[b, : len(ids)] = 1
            # Dedup is only exact if a context masks exactly one position, which
            # is what makes the context string identify the position.
            if context.count(MASK_CHAR) != 1:
                raise ValueError(f"Context does not mask exactly one position: {context}")
            mask_pos.append(context.index(MASK_CHAR))  # no CLS offset

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

        # Preprocess sequences (uppercase, DNA -> RNA)
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

        # Initialize model. The vocabulary is read from the checkpoint directly.
        print(f"Initializing RNAGenesis model ({args.model_name})...")
        vocab = load_vocab(args.model_name)
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name, trust_remote_code=True, torch_dtype=getattr(torch, args.dtype)
        )
        model = model.to(args.device)
        model.eval()

        # Run masked-marginal inference
        print("Running inference...")
        sequence_scores = run_inference(
            model,
            vocab,
            contexts,
            ctx_tasks,
            scores,
            args.device,
            args.batch_size,
            args.max_batch_tokens,
        )

        # Add scores to DataFrame
        score_column = "rnagenesis_score"
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

# python score_rnagenesis_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path rnagenesis_output --model_name /path/to/rnagenesis_checkpoint
