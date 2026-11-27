#!/usr/bin/env python3
"""Score structRFM with the shared masked-marginal engine.

structRFM is a BERT masked language model with an RNA alphabet. The released
Hugging Face checkpoint includes the masked-LM head, and the decoder is tied to
the input embeddings.
"""

import argparse

import torch
from typing_extensions import override

from rnagym.fitness.baselines.masked_lm import MaskedLMAdapter, main
from rnagym.fitness.tasks.model_registry import checkpoint_revision

# Weights that transformers ties rather than loads: the decoder shares the
# input embeddings and the prediction-head bias
TIED_KEYS = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}


class StructRFMAdapter(MaskedLMAdapter):
    """structRFM tokenizer and masked-LM forward pass."""

    name = "structRFM"
    bases = "ACGU"
    score_column = "structrfm_score"
    max_tokens = 514  # position limit including [CLS] and [SEP]
    n_special_tokens = 2  # [CLS] and [SEP]
    n_id = 0

    @staticmethod
    @override
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            dest="model_name",
            type=str,
            default="heqin-zhu/structRFM",
            help="Checkpoint ID or local path",
        )

    @override
    def load(self, args: argparse.Namespace) -> None:
        from transformers import BertForMaskedLM, PreTrainedTokenizerFast

        revision = checkpoint_revision(args.model_name)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            args.model_name, revision=revision
        )
        model, info = BertForMaskedLM.from_pretrained(
            args.model_name,
            revision=revision,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )
        missing = set(info["missing_keys"]) - TIED_KEYS
        if missing or info["unexpected_keys"] or info["mismatched_keys"]:
            raise ValueError(
                f"{self.name} checkpoint did not load cleanly: missing {sorted(missing)}, "
                f"unexpected {info['unexpected_keys']}, "
                f"mismatched {info['mismatched_keys']}"
            )
        decoder = model.get_output_embeddings().weight
        embeddings = model.get_input_embeddings().weight
        if decoder.data_ptr() != embeddings.data_ptr():
            raise ValueError(f"{self.name} decoder is not tied to its embeddings")
        if model.config.max_position_embeddings != self.max_tokens:
            raise ValueError(
                f"{self.name} has {model.config.max_position_embeddings} positions, "
                f"expected {self.max_tokens}"
            )
        self.model = model.to(args.device).eval()
        self.device = args.device

        self.base_ids = {b: tokenizer.convert_tokens_to_ids(b) for b in self.bases}
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        # Any other character is [UNK], as in the authors' tokenizer. The vocabulary
        # also has its own N token, used for an N in context; N is not a scored base
        self.unk_id = tokenizer.unk_token_id
        self.n_id = tokenizer.convert_tokens_to_ids("N")
        if self.n_id in (self.unk_id, self.mask_id, *self.base_ids.values()):
            raise ValueError(f"{self.name} has no distinct N token: {self.n_id}")
        self.prefix_ids = [tokenizer.cls_token_id]
        self.suffix_ids = [tokenizer.sep_token_id]

    @override
    def encode_context(self, context: str, mask_char: str) -> list[int]:
        ids = super().encode_context(context, mask_char)
        offset = len(self.prefix_ids)
        for position, char in enumerate(context):
            if char == "N":
                ids[offset + position] = self.n_id
        return ids

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
    main(StructRFMAdapter())
