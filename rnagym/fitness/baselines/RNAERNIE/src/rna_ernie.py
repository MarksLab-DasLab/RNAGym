"""RNA-ERNIE sequence batching."""

from pathlib import Path

import paddle
from rnagym.fitness.baselines.RNAERNIE.src.dataset_utils import seq2input_ids
from rnagym.fitness.baselines.RNAERNIE.src.tokenizer_nuc import NUCTokenizer


class BatchConverter:
    """Convert named sequences to fixed-length RNA-ERNIE token batches."""

    def __init__(
        self,
        vocab_path: Path,
        batch_size: int = 256,
        max_sequence_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_sequence_length < 3:
            raise ValueError("max_sequence_length must leave room for special tokens")
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = NUCTokenizer(k_mer=1, vocab_file=str(vocab_path))

    def __call__(self, data: list[tuple[str, str]]):
        """Yield names, normalized sequences and padded token tensors."""
        for start in range(0, len(data), self.batch_size):
            batch = data[start : start + self.batch_size]
            names = [name for name, _ in batch]
            sequences = [sequence.upper().replace("U", "T") for _, sequence in batch]
            too_long = [
                name
                for name, sequence in zip(names, sequences)
                if len(sequence) > self.max_sequence_length - 2
            ]
            if too_long:
                raise ValueError(f"Sequences exceed the model limit: {too_long}")
            input_ids = [
                seq2input_ids(sequence, self.tokenizer) for sequence in sequences
            ]
            input_ids = [
                tokens + [0] * (self.max_sequence_length - len(tokens))
                for tokens in input_ids
            ]
            yield names, sequences, paddle.to_tensor(input_ids)
