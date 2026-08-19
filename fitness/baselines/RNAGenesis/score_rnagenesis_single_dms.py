#!/usr/bin/env python3
"""
Score DMS assay sequences with RNAGenesis.

RNAGenesis (https://github.com/zaixizhang/RNAGenesis) is a generalist RNA
foundation model. Its released encoder (https://huggingface.co/Zaixi/RNAGenesis)
is an xTrimoPGLM-style bidirectional transformer pretrained with a masked
language-modelling objective on RNAcentral, exposed as
``xTrimoPGLMForMaskedLM``. It is scored with the shared masked-marginal engine
in ``fitness/baselines/masked_lm``, which offers the four fill strategies
(``wt-fill``, ``mask-fill``, ``mut-fill``, ``match-fill``) that differ in what
the model sees at a variant's other mutated positions.

Two properties of the released checkpoint drive the implementation:

1. The model uses the RNA alphabet. Its vocabulary contains U and no T, so
   sequences are folded to U (the opposite of AIDO.RNA).
2. The shipped tokenizer wrapper cannot be used directly. ``<mask>``, ``<unk>``
   and ``<eos>`` are absent from the vocabulary and receive phantom
   added-token ids past the real vocab, ``mask_token_id`` is None, and
   ``convert_tokens_to_ids`` iterates a string character by character. The
   author's own ``run.py`` depends on that character iteration and adds no
   special tokens, so input ids are built directly from ``tokenizer.model``:
   one id per nucleotide, no CLS or EOS.

The mask token is ``tMASK``, the token-level mask of the xTrimoPGLM family. It
was confirmed empirically: on wild-type ncRNA constructs it gives a lower masked
negative log-likelihood than gMASK or sMASK, and puts over 99.9% of the
predicted mass on A/C/G/U.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from masked_lm import MaskedLMAdapter, main  # noqa: E402

MASK_TOKEN = "tMASK"
UNK_TOKEN = "N"


def load_vocab(model_path: str) -> dict:
    """
    Read the model's own vocabulary file and return a token to id mapping.

    The HuggingFace tokenizer wrapper is bypassed on purpose, see the module
    docstring.
    """
    vocab_file = Path(model_path) / "tokenizer.model"
    if not vocab_file.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_file}. Point --model_name at a "
            f"local copy of the RNAGenesis checkpoint."
        )
    tokens = vocab_file.read_text().splitlines()
    return {token: index for index, token in enumerate(tokens)}


class RNAGenesisAdapter(MaskedLMAdapter):
    """RNAGenesis: RNA alphabet, no special tokens, attention mask."""

    name = "RNAGenesis"
    bases = "ACGU"
    score_column = "rnagenesis_score"
    n_special_tokens = 0  # the reference usage adds none
    default_batch_size = 256
    default_max_batch_tokens = 32768

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model_name",
            type=str,
            required=True,
            help="Local path of the RNAGenesis encoder checkpoint directory, "
            "which must contain tokenizer.model",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float32"],
            help="Torch dtype for the model weights. bfloat16 is the default and "
            "is what the released predictions were produced with",
        )

    def load(self, args):
        from transformers import AutoModelForMaskedLM

        vocab = load_vocab(args.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=getattr(torch, args.dtype),
        )
        self.model = self.model.to(args.device).eval()
        self.device = args.device

        self.base_ids = {b: vocab[b] for b in self.bases}
        self.mask_id = vocab[MASK_TOKEN]
        self.unk_id = vocab[UNK_TOKEN]
        self.pad_id = vocab.get("<pad>", self.unk_id)
        self.prefix_ids = []  # the reference usage adds no special tokens
        self.suffix_ids = []

    def logits_at(self, input_ids, attention_mask, rows, cols):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[rows, cols]


if __name__ == "__main__":
    main(RNAGenesisAdapter())

# python score_rnagenesis_single_dms.py --row_id 0 --ref_sheet reference_sheet.csv --dms_dir_path fitness_processed_assays --output_dir_path rnagenesis_output --model_name /path/to/rnagenesis_checkpoint
