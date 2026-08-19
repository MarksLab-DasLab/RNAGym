#!/usr/bin/env python3
"""
Score DMS assay sequences with RNA-FM.

RNA-FM is scored with the shared masked-marginal engine in
``fitness/baselines/masked_lm``, which offers the four fill strategies
(``wt-fill``, ``mask-fill``, ``mut-fill``, ``match-fill``) that differ in what
the model sees at a variant's other mutated positions. See that package's
docstrings for the formulas and their provenance.

This replaces the earlier ``compute_fitness.py``, which offered two strategies
and whose released predictions match neither of the masked ones. Its
``masked-marginals`` masked ALL of a variant's mutated positions at once in the
WILD-TYPE sequence, which is this package's ``mask-fill``, and its
``wt-marginals`` did not mask at all. The released RNA-FM predictions are
reproduced to floating-point noise by ``wt-marginals`` (Pearson 1.000000, max
abs difference under 2e-5 on the two assays checked) and not by any masked
strategy (Pearson 0.28 to 0.89), even though its run script requested
``masked-marginals``.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masked_lm import MaskedLMAdapter, main  # noqa: E402


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
            help="Path to the RNA-FM_pretrained.pth weights file. Note this differs "
            "from compute_fitness.py's --model_location, which is the cloned RNA-FM "
            "module directory added to sys.path; here the package is expected to be "
            "installed (pip install rna-fm) and only the checkpoint is passed.",
        )

    def load(self, args):
        # The released checkpoint pickles an argparse.Namespace, which torch>=2.6
        # refuses to unpickle under the weights_only default, so that one class is
        # allowlisted rather than disabling the check entirely.
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

# python score_rna_fm_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path rna_fm_output --checkpoint_path RNA-FM_pretrained.pth
