#!/usr/bin/env python3
"""Score RiNALMo with the shared masked-marginal engine.

RiNALMo uses a DNA alphabet, so U is folded to T before token lookup. Its
transformer runs in bfloat16 for flash attention and its masked-LM head is
recomputed in float32 to avoid quantized log-likelihood ratios.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masked_lm import MaskedLMAdapter, main  # noqa: E402


class RiNALMoAdapter(MaskedLMAdapter):
    """RiNALMo: DNA alphabet, ``<cls>`` and ``<eos>``, CUDA only."""

    name = "RiNALMo"
    bases = "ACGT"
    score_column = "logit_scores"
    max_tokens = 1024
    n_special_tokens = 2
    requires_cuda = True  # the giga checkpoint stores flash-attention modules

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--checkpoint_path",
            type=str,
            required=True,
            help="Path to the RiNALMo giga-v1 checkpoint (.pt), e.g. "
            "rinalmo_giga_pretrained.pt from the project's Zenodo record.",
        )

    def load(self, args):
        # The released checkpoint stores the flash-attention module layout, so
        # flash attention must stay enabled
        from rinalmo.config import model_config
        from rinalmo.data.alphabet import Alphabet
        from rinalmo.model.model import RiNALMo

        config = model_config("giga")
        self.model = RiNALMo(config)
        self.model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
        self.model = self.model.to(args.device).eval()
        self.device = args.device

        alphabet = Alphabet(**config["alphabet"])
        self.base_ids = {b: alphabet.get_idx(b) for b in self.bases}
        self.mask_id = alphabet.mask_idx
        self.pad_id = alphabet.pad_idx
        self.unk_id = alphabet.unk_idx
        self.prefix_ids = [alphabet.cls_idx]
        self.suffix_ids = [alphabet.eos_idx]

    def logits_at(self, input_ids, attention_mask, rows, cols):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            representation = self.model(input_ids)["representation"]
        return self.model.lm_mask_head(representation.float())[rows, cols]


if __name__ == "__main__":
    main(RiNALMoAdapter())
