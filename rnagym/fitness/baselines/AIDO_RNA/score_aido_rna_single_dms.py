#!/usr/bin/env python3
"""Score AIDO.RNA with the shared masked-marginal engine.

The adapter uses GenBio AI's ``modelgenerator`` implementation and the model's
DNA alphabet.
"""

import argparse
from pathlib import Path

import torch
from typing_extensions import override

from rnagym.fitness.baselines.masked_lm import MaskedLMAdapter, main
from rnagym.fitness.tasks.model_registry import checkpoint_revision


class AIDORNAAdapter(MaskedLMAdapter):
    """AIDO.RNA tokenizer and masked-LM forward pass."""

    name = "AIDO.RNA"
    bases = "ACGT"
    score_column = "aido_rna_score"
    n_special_tokens = 2  # [CLS] and [SEP]
    default_max_batch_tokens = 49152

    @staticmethod
    @override
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            dest="model_name",
            type=str,
            default="genbio-ai/AIDO.RNA-1.6B",
            help="Checkpoint ID or local path",
        )

    @override
    def load(self, args: argparse.Namespace) -> None:
        # The tokenizer vocabulary ships with modelgenerator
        import modelgenerator
        from modelgenerator.huggingface_models.rnabert import (
            RNABertForMaskedLM,
            RNABertTokenizer,
        )

        if modelgenerator.__file__ is None:
            raise ImportError("modelgenerator has no package location")
        vocab_file = (
            Path(modelgenerator.__file__).parent
            / "huggingface_models/rnabert/vocab.txt"
        )
        tokenizer = RNABertTokenizer(str(vocab_file), version="v2")
        self.model = RNABertForMaskedLM.from_pretrained(
            args.model_name,
            revision=checkpoint_revision(args.model_name),
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(args.device).eval()
        self.device = args.device

        self.base_ids = {b: tokenizer.convert_tokens_to_ids(b) for b in self.bases}
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.unk_id = tokenizer.unk_token_id  # any base outside ACGT, e.g. an N
        self.prefix_ids = [tokenizer.cls_token_id]
        self.suffix_ids = [tokenizer.sep_token_id]

    @override
    def logits_at(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[rows, cols]


if __name__ == "__main__":
    main(AIDORNAAdapter())
