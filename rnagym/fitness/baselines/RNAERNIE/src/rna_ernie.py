"""Single-nucleotide token batches for the released RNA-ERNIE checkpoint."""

from pathlib import Path

import paddle


class BatchConverter:
    """Encode sequences as CLS, nucleotide IDs, SEP and right padding."""

    def __call__(self, data: list[tuple[str, str]]):
        """Yield names, normalized sequences and padded token tensors."""
        for start in range(0, len(data), self.batch_size):
            batch = data[start : start + self.batch_size]
            names = [name for name, _ in batch]
            sequences = [sequence.upper().replace("U", "T") for _, sequence in batch]
            encoded = []
            for name, sequence in zip(names, sequences):
                if len(sequence) > self.max_sequence_length - 2:
                    raise ValueError(f"{name} exceeds the model sequence limit")
                tokens = [
                    self.alphabet["[CLS]"],
                    *(self.alphabet[base] for base in sequence),
                    self.alphabet["[SEP]"],
                ]
                encoded.append(
                    tokens
                    + [self.alphabet["[PAD]"]]
                    * (self.max_sequence_length - len(tokens))
                )
            yield names, sequences, paddle.to_tensor(encoded)

    def __init__(
        self, vocab_path: Path, batch_size: int = 256, max_sequence_length: int = 512
    ) -> None:
        if batch_size < 1 or max_sequence_length < 3:
            raise ValueError(
                "Batch size must be positive and sequence length must allow special tokens"
            )
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.alphabet = {
            token: index
            for index, token in enumerate(vocab_path.read_text().splitlines())
        }
        required = {"[CLS]", "[SEP]", "[PAD]", "A", "C", "G", "T"}
        if not required <= self.alphabet.keys():
            raise ValueError(
                f"Vocabulary is missing tokens: {sorted(required - self.alphabet.keys())}"
            )
