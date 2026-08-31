#!/usr/bin/env python3
"""Score RNA-FM with the shared masked-marginal engine."""

import torch
from rnagym.fitness.baselines.masked_lm import MaskedLMAdapter, main


class RNAFMAdapter(MaskedLMAdapter):
    """RNA-FM: RNA alphabet, ``<cls>`` and ``<eos>``, no attention mask."""

    name = "RNA-FM"
    bases = "ACGU"
    score_column = "RNA_FM_scores"
    max_tokens = 1024  # position limit including <cls> and <eos>
    n_special_tokens = 2

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--checkpoint_path",
            type=str,
            required=True,
            help="Path to the RNA-FM_pretrained.pth weights file. The rna-fm package "
            "must be installed in the model environment",
        )

    def load(self, args):
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

    def logits_at(self, input_ids, attention_mask, rows, cols):
        return self.model(input_ids)["logits"][rows, cols]


if __name__ == "__main__":
    main(RNAFMAdapter())
