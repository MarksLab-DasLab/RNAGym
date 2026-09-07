#!/usr/bin/env python3
"""Score DMS assays using Nucleotide Transformer v3's pretrained MLM head.

NTv3 requires lengths divisible by its U-Net downsampling factor. Upstream
recommends padding shorter inputs with N, because the model was not trained on
padding tokens. This benchmark applies its shared masked-marginal estimator,
not the authors' supervised variant-effect method.
"""

import argparse

import torch
from typing_extensions import override

from rnagym.fitness.baselines.masked_lm import MaskedLMAdapter, main
from rnagym.fitness.tasks.model_registry import checkpoint_revision

# Tokenizer, configuration and model code live in InstaDeepAI/ntv3_base_model
CODE_REVISION = "0ecff3637f0d3ba5b686d1095083218157c2ca34"


class NTv3Adapter(MaskedLMAdapter):
    """Single-base NTv3 adapter with architecture-required N padding."""

    name = "Nucleotide Transformer v3"
    bases = "ACGT"
    score_column = "ntv3_score"
    context_pad_char = "N"
    default_batch_size = 128

    @staticmethod
    @override
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            dest="model_name",
            default="InstaDeepAI/NTv3_650M_pre",
            help="Pretrained checkpoint ID or local path",
        )

    @override
    def context_length_for(self, length: int) -> int:
        """Round up to the checkpoint's U-Net block size."""
        multiple = 2**self.num_downsamples
        return ((length + multiple - 1) // multiple) * multiple

    @override
    def load(self, args: argparse.Namespace) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        revision = checkpoint_revision(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            revision=revision,
            code_revision=CODE_REVISION,
            trust_remote_code=True,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            args.model_name,
            revision=revision,
            code_revision=CODE_REVISION,
            trust_remote_code=True,
        )
        self.model = self.model.to(args.device).eval()
        self.device = args.device
        self.num_downsamples = int(self.model.config.num_downsamples)

        vocab = tokenizer.get_vocab()
        self.base_ids = {base: vocab[base] for base in self.bases if base in vocab}
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.unk_id = vocab["N"]

    @override
    def logits_at(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        if (input_ids == self.pad_id).any():
            raise ValueError(
                f"{self.name} ignores the attention mask, but a padding token "
                "reached the model"
            )
        return self.model(input_ids=input_ids).logits[rows, cols]


if __name__ == "__main__":
    main(NTv3Adapter())
