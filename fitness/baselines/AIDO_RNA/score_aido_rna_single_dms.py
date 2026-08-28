#!/usr/bin/env python3
"""Score AIDO.RNA with the shared masked-marginal engine.

The adapter uses GenBio AI's ``modelgenerator`` implementation and the model's
DNA alphabet.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masked_lm import MaskedLMAdapter, main  # noqa: E402


class AIDORNAAdapter(MaskedLMAdapter):
    """AIDO.RNA tokenizer and masked-LM forward pass."""

    name = "AIDO.RNA"
    bases = "ACGT"
    score_column = "aido_rna_score"
    n_special_tokens = 2  # [CLS] and [SEP]
    default_max_batch_tokens = 49152

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model_name",
            type=str,
            default="genbio-ai/AIDO.RNA-1.6B",
            help="HuggingFace model id or local path of the AIDO.RNA checkpoint "
            "(default: genbio-ai/AIDO.RNA-1.6B)",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float32"],
            help="Torch dtype for the model weights. bfloat16 is the default and "
            "is what the released predictions were produced with. fp32 changed "
            "the aptamer Spearman by at most 0.0006 when it was checked",
        )

    def load(self, args):
        from modelgenerator.huggingface_models.rnabert import (
            RNABertForMaskedLM,
            RNABertTokenizer,
        )

        # The tokenizer vocabulary ships with modelgenerator
        vocab_file = os.path.join(
            os.path.dirname(__import__("modelgenerator").__file__),
            "huggingface_models",
            "rnabert",
            "vocab.txt",
        )
        tokenizer = RNABertTokenizer(vocab_file, version="v2")
        self.model = RNABertForMaskedLM.from_pretrained(
            args.model_name, torch_dtype=getattr(torch, args.dtype)
        )
        self.model = self.model.to(args.device).eval()
        self.device = args.device

        self.base_ids = {b: tokenizer.convert_tokens_to_ids(b) for b in self.bases}
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.unk_id = tokenizer.unk_token_id  # any base outside ACGT, e.g. an N
        self.prefix_ids = [tokenizer.cls_token_id]
        self.suffix_ids = [tokenizer.sep_token_id]

    def logits_at(self, input_ids, attention_mask, rows, cols):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[rows, cols]


if __name__ == "__main__":
    main(AIDORNAAdapter())
