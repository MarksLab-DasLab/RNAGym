#!/usr/bin/env python3
"""
Score DMS assay sequences with Orthrus.

Orthrus (https://github.com/bowang-lab/Orthrus) is a Mamba-based RNA foundation
model. Its contrastive checkpoints produce embeddings only, but
``antichronology/orthrus-mlm-6-track`` is dual-objective pretrained, contrastive
plus masked LM, and exposes ``predict_tokens`` returning per-position logits over
[A, C, G, T]. That checkpoint's own documentation describes the method as
variant scoring and specifies the masking convention:

    "Positions to score should be masked (nucleotide channels set to zero)
     before calling."

So zeroing a position's nucleotide channels is Orthrus's masking operation
rather than a stand-in for a mask token, and the masked-marginal fill strategies
apply to it the same way they apply to the token-based models. This scorer
therefore offers the same ``--strategies`` as the other masked language models;
see fitness/baselines/masked_lm/strategies.py for the formulas.

Orthrus is a 6-track model: 4 one-hot nucleotide channels plus a CDS and a
splice channel derived from a transcript's exon and CDS structure. DMS
constructs are bare sequences with no such annotation, so those 2 channels are
zero throughout, for the wild type and the variant alike. That is a limitation
of applying the model to this data, not of the masking.

The shared engine works on token ids, which Orthrus does not have. It is given
integer base codes instead, 0 to 3 for A, C, G, T with 4 for a masked position,
and ``logits_at`` expands those into the 6-track float tensor the model expects.
That keeps Orthrus on the same context construction, deduplication, batching and
accumulation as every other masked model, with no special case in the engine.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masked_lm import MaskedLMAdapter, main  # noqa: E402


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
    # distinct integers; logits_at turns them back into channels.
    MASK_CODE = 4
    PAD_CODE = 5

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model_name",
            type=str,
            default="antichronology/orthrus-mlm-6-track",
            help="Orthrus checkpoint with an MLM head (default: "
            "antichronology/orthrus-mlm-6-track). The contrastive checkpoints "
            "raise NotImplementedError from predict_tokens",
        )

    def load(self, args):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
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

    def logits_at(self, input_ids, attention_mask, rows, cols):
        """
        Expand base codes into Orthrus's 6-track input and read the MLM head.

        A masked position is left as an all-zero column, which is the masking
        convention the checkpoint documents. Padding is zero as well, and the
        real lengths are passed through so the backbone ignores it.
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

# python score_orthrus_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path orthrus_output
