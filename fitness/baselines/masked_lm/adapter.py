"""
The model-specific half of masked-marginal scoring.

An adapter supplies an alphabet, a tokenization, and one forward pass. The
engine supplies everything else, so adding a masked RNA language model to the
benchmark means writing the subclass below and nothing more.
"""


class MaskedLMAdapter:
    """
    Interface between a masked language model and the scoring engine.

    Class attributes:
        name: model name, for log messages.
        bases: the four bases in the model's own alphabet, ``"ACGU"`` or
            ``"ACGT"``. Assay sequences and mutation strings are folded to it.
        score_column: stem of the output column, kept per model so that the
            prediction files stay drop-in replacements for the released ones.
        max_tokens: position limit including special tokens, or None if the
            model has no fixed limit. When set, a ``--max_tokens`` flag is
            offered and contexts longer than the limit are windowed.
        default_batch_size, default_max_batch_tokens: batching defaults.

    Instance attributes, set by ``load``:
        device, prefix_ids, suffix_ids, pad_id, mask_id, unk_id, base_ids.
    """

    name = "masked LM"
    bases = "ACGU"
    score_column = "score"
    max_tokens = None
    default_batch_size = 512
    default_max_batch_tokens = 65536
    # Whether contexts of different lengths may share a padded batch. All four
    # current models are scored on fixed-length constructs, and RNA-FM and
    # RiNALMo are run without an explicit attention mask, so this is False
    # unless a model is known to be padding invariant.
    allows_mixed_length_batches = False
    requires_cuda = False
    # Number of special tokens the encoding adds, declared statically so that
    # the window-limit guard can run before the model is loaded. Checked against
    # the loaded tokenizer in ``check_alphabet``.
    n_special_tokens = 0
    # Whether context positions map to token positions by a constant shift, which
    # is what lets the engine gather without calling ``token_position`` per term.
    # An adapter that overrides ``token_position`` non-linearly must clear this.
    constant_token_offset = True

    prefix_ids = ()
    suffix_ids = ()
    pad_id = 0
    mask_id = 0
    unk_id = 0
    base_ids = {}
    device = "cpu"

    @staticmethod
    def add_arguments(parser):
        """Add the model's own command line arguments, such as a checkpoint."""
        raise NotImplementedError

    def load(self, args):
        """Construct the model and tokenizer and fill in the token ids."""
        raise NotImplementedError

    def logits_at(self, input_ids, attention_mask, rows, cols):
        """
        Run one batch and return the raw logits at the requested positions.

        Args:
            input_ids: LongTensor ``(B, L)`` already on the device, padded with
                ``pad_id`` and carrying the model's special tokens.
            attention_mask: LongTensor ``(B, L)``, 1 on real tokens. Models that
                do not take an attention mask ignore it.
            rows, cols: LongTensor ``(N,)`` each, indexing the positions to read.
                A context can appear more than once, once per masked position.

        Returns:
            Tensor ``(N, vocab)`` of logits. The engine casts to float32 and
            takes the log-softmax, so returning raw logits keeps the numerics
            identical across models.
        """
        raise NotImplementedError

    def canonicalize_sequence(self, sequence: str) -> str:
        """
        Uppercase, strip whitespace, and fold to the model's own alphabet.

        RNA-FM and RNAGenesis read U, RiNALMo and AIDO.RNA read T, and a
        mismatch puts near-zero probability on every allele, so the folding
        direction is derived from ``bases`` rather than written out per model.
        Symbols outside the alphabet, such as N, are left alone and are encoded
        as the unknown token.
        """
        sequence = str(sequence).strip().upper()
        if "T" in self.bases:
            return sequence.replace("U", "T")
        return sequence.replace("T", "U")

    def encode_context(self, context: str, mask_char: str) -> list:
        """
        Turn a masked context string into model input ids.

        The default is one token per character between the model's leading and
        trailing special tokens, which is what all four single-nucleotide models
        do. A model whose tokenizer does anything else overrides this together
        with ``token_position``.
        """
        ids = list(self.prefix_ids)
        for char in context:
            ids.append(self.mask_id if char == mask_char else self.base_ids.get(char, self.unk_id))
        ids.extend(self.suffix_ids)
        return ids

    def token_position(self, context_position: int) -> int:
        """
        Map a position in the context string to a position in the input ids.

        With one token per nucleotide this is just the number of leading special
        tokens, but the engine goes through this method rather than assuming it,
        since an off-by-one here reads a neighbouring nucleotide's distribution
        and still produces plausible scores.
        """
        return len(self.prefix_ids) + context_position

    def check_alphabet(self):
        """Fail loudly if the tokenizer does not cover the model's alphabet."""
        n_special = len(self.prefix_ids) + len(self.suffix_ids)
        if n_special != self.n_special_tokens:
            raise ValueError(
                f"{self.name} declares {self.n_special_tokens} special tokens but "
                f"its loaded tokenizer adds {n_special}. The window-limit guard "
                "runs before loading and would use the wrong budget"
            )
        missing = [b for b in self.bases if b not in self.base_ids]
        if missing:
            raise ValueError(f"Tokenizer does not cover {missing} of {self.bases}")
        if self.unk_id in self.base_ids.values():
            raise ValueError(
                f"A base maps to the unknown token id {self.unk_id}: {self.base_ids}. "
                "This is the RiNALMo lookup bug; the released RiNALMo scores carry it."
            )
        if self.mask_id in self.base_ids.values():
            raise ValueError(f"A base maps to the mask token id: {self.base_ids}")
