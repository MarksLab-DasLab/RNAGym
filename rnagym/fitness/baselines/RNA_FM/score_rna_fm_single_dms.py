#!/usr/bin/env python3
"""Score RNA-FM with the shared masked-marginal engine."""

import argparse

import torch
from typing_extensions import override

from rnagym.fitness.baselines.masked_lm import MaskedLMAdapter, main


class RNAFMAdapter(MaskedLMAdapter):
    """RNA-FM: RNA alphabet, ``<cls>`` and ``<eos>``, no attention mask."""

    name = "RNA-FM"
    bases = "ACGU"
    score_column = "RNA_FM_scores"
    max_tokens = 1024  # position limit including <cls> and <eos>
    n_special_tokens = 2

    @staticmethod
    @override
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint",
            dest="checkpoint_path",
            type=str,
            required=True,
            help="RNA-FM weights (.pth)",
        )

    @override
    def load(self, args: argparse.Namespace) -> None:
        # The released checkpoint pickles an argparse.Namespace, which torch>=2.6
        # refuses to unpickle under the weights_only default, so that one class is
        # allowlisted rather than disabling the check entirely
        import argparse as _argparse

        torch.serialization.add_safe_globals([_argparse.Namespace])
        import fm

        self.model, alphabet = fm.pretrained.rna_fm_t12(args.checkpoint_path)
        self.model = self.model.to(args.device).eval()
        self.device = args.device
        self.base_ids = {b: alphabet.get_idx(b) for b in self.bases}
        self.mask_id = alphabet.mask_idx
        self.pad_id = alphabet.padding_idx
        self.unk_id = alphabet.get_idx("N")  # any base outside ACGU
        self.prefix_ids = [alphabet.cls_idx]
        self.suffix_ids = [alphabet.eos_idx]

    @override
    def logits_at(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids)["logits"][rows, cols]


if __name__ == "__main__":
    main(RNAFMAdapter())
