#!/usr/bin/env python3
"""Score RNAGenesis with the shared masked-marginal engine.

The checkpoint uses an RNA alphabet, one token per nucleotide, no special
tokens, and ``tMASK``. Its vocabulary is read directly from ``tokenizer.model``
because the shipped tokenizer does not expose the model's mask ID.
"""

from pathlib import Path

import torch
from rnagym.fitness.baselines.masked_lm import MaskedLMAdapter, main

MASK_TOKEN = "tMASK"
UNK_TOKEN = "N"


def load_vocab(model_path: Path) -> dict:
    """
    Read the model's own vocabulary file and return a token to id mapping.

    The HuggingFace tokenizer wrapper is bypassed on purpose, see the module
    docstring.
    """
    vocab_file = model_path / "tokenizer.model"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    tokens = vocab_file.read_text().splitlines()
    return {token: index for index, token in enumerate(tokens)}


class RNAGenesisAdapter(MaskedLMAdapter):
    """RNAGenesis: RNA alphabet, no special tokens, attention mask."""

    name = "RNAGenesis"
    bases = "ACGU"
    score_column = "rnagenesis_score"
    n_special_tokens = 0  # the reference usage adds none
    default_batch_size = 256
    default_max_batch_tokens = 32768

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model_name",
            type=Path,
            required=True,
            help="Local RNAGenesis checkpoint directory",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float32"],
            help="Torch dtype for the model weights. bfloat16 is the default and "
            "is what the released predictions were produced with",
        )

    def load(self, args):
        from transformers import AutoModelForMaskedLM

        vocab = load_vocab(args.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=getattr(torch, args.dtype),
        )
        self.model = self.model.to(args.device).eval()
        self.device = args.device

        self.base_ids = {b: vocab[b] for b in self.bases}
        self.mask_id = vocab[MASK_TOKEN]
        self.unk_id = vocab[UNK_TOKEN]
        self.pad_id = vocab.get("<pad>", self.unk_id)
        self.prefix_ids = []  # the reference usage adds no special tokens
        self.suffix_ids = []

    def logits_at(self, input_ids, attention_mask, rows, cols):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[rows, cols]


if __name__ == "__main__":
    main(RNAGenesisAdapter())
