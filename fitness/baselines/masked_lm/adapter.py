"""Model interface for the shared masked-marginal engine."""


class MaskedLMAdapter:
    """Supply model tokenization and logits to the shared engine."""

    name = "masked LM"
    bases = "ACGU"
    score_column = "score"
    max_tokens = None
    default_batch_size = 512
    default_max_batch_tokens = 65536
    # Whether contexts of different lengths may share a padded batch. Current
    # models are scored on fixed-length constructs, and RNA-FM and RiNALMo are
    # run without an explicit attention mask, so this is False unless a model
    # is known to be padding invariant
    allows_mixed_length_batches = False
    requires_cuda = False
    # Number of special tokens the encoding adds, declared statically so that
    # the window-limit guard can run before the model is loaded. Checked against
    # the loaded tokenizer in ``check_alphabet``
    n_special_tokens = 0
    # Whether context positions map to token positions by a constant shift, which
    # is what lets the engine gather without calling ``token_position`` per term
    # An adapter that overrides ``token_position`` non-linearly must clear this
    constant_token_offset = True

    prefix_ids = ()
    suffix_ids = ()
    pad_id = 0
    mask_id = 0
    unk_id = 0
    base_ids = None
    device = "cpu"

    @staticmethod
    def add_arguments(parser):
        """Add the model's own command line arguments, such as a checkpoint."""
        raise NotImplementedError

    def load(self, args):
        """Construct the model and tokenizer and fill in the token ids."""
        raise NotImplementedError

    def logits_at(self, input_ids, attention_mask, rows, cols):
        """Return raw logits of shape ``(positions, vocabulary)``."""
        raise NotImplementedError

    def canonicalize_sequence(self, sequence: str) -> str:
        """Normalize whitespace, case, and the model's T or U alphabet."""
        sequence = str(sequence).strip().upper()
        if "T" in self.bases:
            return sequence.replace("U", "T")
        return sequence.replace("T", "U")

    def encode_context(self, context: str, mask_char: str) -> list:
        """Encode one token per context character plus special tokens."""
        ids = list(self.prefix_ids)
        for char in context:
            token_id = (
                self.mask_id
                if char == mask_char
                else self.base_ids.get(char, self.unk_id)
            )
            ids.append(token_id)
        ids.extend(self.suffix_ids)
        return ids

    def token_position(self, context_position: int) -> int:
        """Map a context coordinate to an input-token coordinate."""
        return len(self.prefix_ids) + context_position

    def check_alphabet(self):
        """Fail loudly if the tokenizer does not cover the model's alphabet."""
        if not isinstance(self.base_ids, dict):
            raise ValueError(f"{self.name} did not load its base token ids")
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
        if len(set(self.base_ids.values())) != len(self.base_ids):
            raise ValueError(f"Bases do not have distinct token ids: {self.base_ids}")
        if self.unk_id in self.base_ids.values():
            raise ValueError(
                f"A base maps to the unknown token id {self.unk_id}: {self.base_ids}"
            )
        if self.mask_id in self.base_ids.values():
            raise ValueError(f"A base maps to the mask token id: {self.base_ids}")
