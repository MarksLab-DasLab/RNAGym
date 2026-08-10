#!/usr/bin/env python3
"""
Script to run RiNALMo inference on DMS assay sequences.
Takes a reference sheet and row ID to process specific assays.

This rescores RiNALMo with the masked-marginal convention used by the other
masked RNA language-model baselines (Orthrus, AIDO.RNA, RNAGenesis, and the
rescored RNA-FM):

    score(variant) = sum_i [ log P(mut_i | variant context, pos_i masked)
                             - log P(wt_i  | variant context, pos_i masked) ]

Each mutated position is masked one at a time in the variant's OWN sequence, so
the remaining mutations of a multi-mutant stay in the context.

This replaces the earlier ``compute_fitness.py``, which already masked one
position at a time in the variant's own sequence but read the wrong logit for the
mutant base. RiNALMo's alphabet is DNA-based (A C G T), and while
``Alphabet.encode`` folds U to T internally, ``Alphabet.get_idx('U')`` returns
<unk>. That script looks the mutant base up in a U-form sequence and the
wild-type base up in a T-form sequence, so every mutation to U scores against
the <unk> logit instead of the T logit. Emulating that lookup reproduces the
published predictions much better (Pearson 0.83 to 0.84) than any correct
implementation (0.13 to 0.27), which is evidence that the published numbers
carry it, though the match is not exact so at least one other difference
remains. That script also stripped N from the wild type inside the scoring loop
without adjusting coordinates, which would shift positions on any construct
containing N. This script does the same masking with correct T-form lookups on
both sides.

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

# RiNALMo uses the DNA alphabet: A, C, G, T. Alphabet.encode folds U to T.
BASES = "ACGT"
MASK_CHAR = "#"


def preprocess_sequence(sequence: str) -> str:
    """
    Preprocess an RNA/DNA sequence for RiNALMo:
    - Convert to uppercase
    - Convert RNA (U) to DNA (T), matching the model's alphabet
    - Remove any whitespace
    """
    return sequence.strip().upper().replace("U", "T")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RiNALMo (masked LM) inference on DMS assay sequences."
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
        default="cuda:0",
        help="CUDA device to run inference on (default: cuda:0). RiNALMo's giga "
        "checkpoint stores the flash-attention module layout, which is CUDA only, "
        "so there is no working CPU path.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the RiNALMo giga-v1 checkpoint (.pt), e.g. "
        "rinalmo_giga_pretrained.pt from the project's Zenodo record.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Maximum masked contexts scored per forward pass (default: 512)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="RiNALMo position limit including <cls> and <eos>. Contexts longer "
        "than this are windowed around the scored position (default: 1024)",
    )
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=65536,
        help="Cap on batch_size x sequence length per forward pass, so that long "
        "assays automatically use a smaller batch (default: 65536)",
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
        wt_base = token[0].upper().replace("U", "T")
        mut_base = token[-1].upper().replace("U", "T")
        pos = int(token[1:-1]) - 1  # 1-based -> 0-based
        if wt_base not in BASES or mut_base not in BASES:
            raise ValueError(f"Unsupported mutation token: {token}")
        mutations.append((pos, wt_base, mut_base))
    return mutations


def window_context(context: str, max_tokens: int) -> str:
    """
    Trim a masked context to fit RiNALMo's position limit, centred on the mask.

    RiNALMo accepts at most ``max_tokens`` tokens including <cls> and <eos>. The
    ncRNA assays are all far shorter than that, but the mRNA-coding constructs
    reach several thousand bases, so a window centred on the scored position is
    taken, mirroring the windowing in this directory's compute_fitness.py.
    Mutations that fall outside the window are lost from the context, which is
    inherent to windowing.
    """
    budget = max_tokens - 2  # room for <cls> and <eos>
    if len(context) <= budget:
        return context
    pos = context.index(MASK_CHAR)
    start = max(0, pos - budget // 2)
    end = min(len(context), start + budget)
    start = max(0, end - budget)
    return context[start:end]


def build_masking_tasks(mutants: list, sequences: list, max_tokens: int) -> tuple:
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
                context = window_context(
                    seq[:pos] + MASK_CHAR + seq[pos + 1 :], max_tokens
                )
                row_tasks.append((context, wt_base, mut_base))
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
    alphabet,
    contexts: list,
    ctx_tasks: list,
    scores: np.ndarray,
    device: str,
    batch_size: int,
    max_batch_tokens: int,
) -> np.ndarray:
    """
    Score masked contexts with RiNALMo's masked LM head and accumulate the
    per-position log-likelihood ratios into each variant's score.

    Token layout follows RiNALMo's Alphabet.encode: ``<cls>`` + sequence +
    ``<eos>``, so the masked position index is offset by one.
    """
    base_ids = {b: alphabet.get_idx(b) for b in BASES}
    if alphabet.unk_idx in base_ids.values():
        raise ValueError(f"Alphabet does not cover {BASES}: {base_ids}")
    mask_id = alphabet.mask_idx
    pad_id = alphabet.pad_idx
    cls_id = alphabet.cls_idx
    eos_id = alphabet.eos_idx
    unk_id = alphabet.get_idx("N")  # any base outside ACGT

    seq_len = max(len(c) for c in contexts)
    batch_size = max(1, min(batch_size, max_batch_tokens // (seq_len + 2)))
    print(f"Using batch size {batch_size} for sequence length {seq_len}")

    for start in tqdm(
        range(0, len(contexts), batch_size), desc="Scoring", unit="batch"
    ):
        batch = contexts[start : start + batch_size]
        max_len = max(len(c) for c in batch) + 2  # <cls> ... <eos>

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask_pos = []
        for b, context in enumerate(batch):
            ids = [cls_id]
            for ch in context:
                ids.append(mask_id if ch == MASK_CHAR else base_ids.get(ch, unk_id))
            ids.append(eos_id)
            input_ids[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            # Dedup is only exact if a context masks exactly one position, which
            # is what makes the context string identify the position.
            if context.count(MASK_CHAR) != 1:
                raise ValueError(f"Context does not mask exactly one position: {context}")
            mask_pos.append(context.index(MASK_CHAR) + 1)  # +1 for <cls>

        input_ids = input_ids.to(device)
        rows = torch.arange(len(batch), device=device)
        cols = torch.tensor(mask_pos, device=device)

        # The transformer body must run in bfloat16 because the checkpoint's
        # flash-attention kernels accept only fp16/bf16, but bf16 logits are too
        # coarse for this score: the model's logits are large enough that bf16
        # spacing quantises log-ratio differences onto multiples of ~0.125, tying
        # ~8% of single mutants at exactly 0. So the masked LM head is recomputed
        # in fp32 from the bf16 representation, which keeps the ranking usable.
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                representation = model(input_ids)["representation"]
            logits = model.lm_mask_head(representation.float())
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
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError(
                "RiNALMo requires a CUDA device: the giga checkpoint stores the "
                "flash-attention module layout, which has no CPU implementation."
            )

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
            dms_df["mutant"].tolist(), sequences, args.max_tokens
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

        # Initialize model. The released checkpoint stores the flash-attention
        # module layout, so flash attention must stay enabled.
        print(f"Initializing RiNALMo model ({args.checkpoint_path})...")
        from rinalmo.config import model_config
        from rinalmo.data.alphabet import Alphabet
        from rinalmo.model.model import RiNALMo

        config = model_config("giga")
        model = RiNALMo(config)
        model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
        alphabet = Alphabet(**config["alphabet"])
        model = model.to(args.device)
        model.eval()

        # Run masked-marginal inference
        print("Running inference...")
        sequence_scores = run_inference(
            model,
            alphabet,
            contexts,
            ctx_tasks,
            scores,
            args.device,
            args.batch_size,
            args.max_batch_tokens,
        )

        # Add scores to DataFrame. The column name matches the original RiNALMo
        # baseline so these predictions are a drop-in replacement.
        score_column = "logit_scores"
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

# python score_rinalmo_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path rinalmo_output --checkpoint_path giga-v1.pt
