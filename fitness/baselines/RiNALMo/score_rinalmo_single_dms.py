#!/usr/bin/env python3
"""
Score DMS assay sequences with RiNALMo.

RiNALMo is scored with the shared masked-marginal engine in
``fitness/baselines/masked_lm``, which offers the four fill strategies
(``wt-fill``, ``mask-fill``, ``mut-fill``, ``match-fill``) that differ in what
the model sees at a variant's other mutated positions.

This replaces the earlier ``compute_fitness.py``, which already masked one
position at a time in the variant's own sequence but read the wrong logit for
the mutant base. RiNALMo's alphabet is DNA-based (A C G T), and while
``Alphabet.encode`` folds U to T internally, ``Alphabet.get_idx('U')`` returns
<unk>. That script looks the mutant base up in a U-form sequence and the
wild-type base up in a T-form sequence, so every mutation to U scores against
the <unk> logit instead of the T logit. Emulating that lookup reproduces the
published predictions much better (Pearson 0.83 to 0.84) than any correct
implementation (0.13 to 0.27), which is evidence that the published numbers
carry it, though the match is not exact so at least one other difference
remains. That script also stripped N from the wild type inside the scoring loop
without adjusting coordinates, which would shift positions on any construct
containing N. The shared engine's ``check_alphabet`` refuses to run if any base
maps to the unknown token, so this class of bug cannot recur silently.

Precision note: the transformer body must run in bfloat16 because the
checkpoint's flash-attention kernels accept only fp16 or bf16, but bf16 logits
are too coarse for this score. The model's logits are large enough that bf16
spacing quantises log-ratio differences onto multiples of about 0.125, tying
about 8% of single mutants at exactly 0. The masked LM head is therefore
recomputed in fp32 from the bf16 representation, which keeps the ranking usable.
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
        # flash attention must stay enabled.
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

# python score_rinalmo_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path rinalmo_output --checkpoint_path rinalmo_giga_pretrained.pt
