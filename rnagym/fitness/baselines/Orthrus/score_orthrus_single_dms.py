#!/usr/bin/env python3
"""
Score DMS assay sequences with Orthrus's official MLM predictor.

The upstream model card defines variant scoring with ``predict_tokens``. This
adapter calls that interface while the shared engine supplies batching and the
multi-mutant fill contexts. DMS constructs have no CDS or splice annotation, so
Orthrus's two optional annotation channels remain zero.

https://huggingface.co/antichronology/orthrus-mlm-6-track/blob/5f0dc87d51065035fc28e71972c69f9c84f4deae/orthrus_hf.py#L290-L325
"""

import argparse

import torch
from typing_extensions import override

from rnagym.fitness.baselines.masked_lm import MaskedLMAdapter, main


class OrthrusAdapter(MaskedLMAdapter):
    """Orthrus MLM: 6-track continuous input, masked by zeroing channels."""

    name = "Orthrus"
    bases = "ACGT"  # seq_to_oh ordering, and predict_tokens' logit ordering
    score_column = "orthrus_score"
    n_special_tokens = 0
    requires_cuda = True  # the Mamba backbone is CUDA only
    default_batch_size = 64
    default_max_batch_tokens = 65536

    # Base codes standing in for token ids. The engine only needs these to be
    # distinct integers and logits_at turns them back into channels
    MASK_CODE = 4
    PAD_CODE = 5

    @staticmethod
    @override
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model",
            dest="model_name",
            type=str,
            default="antichronology/orthrus-mlm-6-track",
            help="MLM checkpoint ID or local path",
        )

    @override
    def load(self, args: argparse.Namespace) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        self.model = self.model.to(args.device).eval()
        self.device = args.device
        if getattr(self.model, "sequence_head", None) is None:
            raise ValueError(
                f"{args.model_name} has no MLM head, so it cannot score variants. "
                "Use an orthrus-mlm-* checkpoint"
            )
        self.base_ids = {b: i for i, b in enumerate(self.bases)}
        self.mask_id = self.MASK_CODE
        self.pad_id = self.PAD_CODE
        self.unk_id = self.MASK_CODE  # a base outside ACGT is left unset, as a mask is
        self.prefix_ids = []
        self.suffix_ids = []

    @override
    def logits_at(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert base codes to six-track inputs for ``predict_tokens``.

        The nucleotide channels hold one-hot bases. Annotation channels stay
        zero because the DMS assays do not supply them.
        """
        batch, length = input_ids.shape
        x = torch.zeros(batch, length, 6, dtype=torch.float32, device=input_ids.device)
        nucleotide = input_ids < len(self.bases)
        positions = nucleotide.nonzero(as_tuple=True)
        x[positions[0], positions[1], input_ids[nucleotide]] = 1.0
        lengths = attention_mask.sum(dim=1)
        return self.model.predict_tokens(x, lengths, channel_last=True)[rows, cols]


if __name__ == "__main__":
    main(OrthrusAdapter())
