"""Score one fitness assay with RNA-ERNIE."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import paddle
import polars as pl
from paddlenlp.transformers import ErnieForMaskedLM
from rnagym.config import ConfigFitness
from rnagym.fitness.baselines.RNAERNIE.src.rna_ernie import BatchConverter

BASES = "ACGT"
MUTATION_PATTERN = re.compile(r"^[ACGT]\d+[ACGT](?:,[ACGT]\d+[ACGT])*$")
SCORE_COLUMN = "Mutation_Scores"


def normalize_sequence(sequence: str | None) -> str:
    """Return one validated sequence in RNA-ERNIE's DNA alphabet."""
    if sequence is None or not sequence.strip():
        raise ValueError("Sequence is missing or empty")
    normalized = sequence.upper().replace("U", "T")
    invalid = sorted(set(normalized) - set(BASES))
    if invalid:
        raise ValueError(f"Sequence contains unsupported bases: {invalid}")
    return normalized


def parse_mutations(mutant: str | None, wild_type: str) -> list[tuple[int, str, str]]:
    """Parse substitutions and validate them against the wild type."""
    if mutant is None:
        raise ValueError("Mutation is missing")
    normalized = mutant.upper().replace("U", "T").replace(" ", "")
    if MUTATION_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"Invalid mutation string: {mutant!r}")

    mutations = []
    positions = set()
    for token in normalized.split(","):
        position = int(token[1:-1]) - 1
        source, target = token[0], token[-1]
        if position in positions:
            raise ValueError(f"Mutation string repeats position {position + 1}")
        if position < 0 or position >= len(wild_type):
            raise ValueError(
                f"Mutation position {position + 1} is outside a "
                f"{len(wild_type)} nt construct"
            )
        if wild_type[position] != source:
            raise ValueError(
                f"Mutation {mutant!r} expects {source} at position {position + 1}, "
                f"but the reference has {wild_type[position]}"
            )
        positions.add(position)
        mutations.append((position, source, target))
    return mutations


def score_assay(
    model: ErnieForMaskedLM,
    converter: BatchConverter,
    assay_file: Path,
    wild_type: str,
) -> pl.DataFrame:
    """Score every non-wild-type row in one assay."""
    _, _, input_ids = next(iter(converter([("wild_type", wild_type)])))
    with paddle.no_grad():
        logits = model(input_ids)
        probabilities = paddle.nn.functional.softmax(logits, axis=-1)[0].numpy()

    assay = pl.read_csv(assay_file)
    required_columns = {"mutant", "sequence"}
    missing_columns = sorted(required_columns - set(assay.columns))
    if missing_columns:
        raise ValueError(f"{assay_file} is missing columns: {missing_columns}")
    assay = assay.filter(pl.col("mutant").is_not_null())
    if assay.is_empty():
        raise ValueError(f"{assay_file} has no variants")

    alphabet = converter.tokenizer.vocab.token_to_idx
    scores = []
    normalized_mutants = []
    for row, (mutant, sequence) in enumerate(
        assay.select("mutant", "sequence").iter_rows()
    ):
        mutations = parse_mutations(mutant, wild_type)
        variant = list(wild_type)
        score = 0.0
        for position, source, target in mutations:
            variant[position] = target
            score += float(
                probabilities[position + 1, alphabet[target]]
                - probabilities[position + 1, alphabet[source]]
            )
        if normalize_sequence(sequence) != "".join(variant):
            raise ValueError(f"{assay_file} row {row} sequence disagrees with mutant")
        scores.append(score)
        normalized_mutants.append(
            ",".join(
                f"{source}{position + 1}{target}"
                for position, source, target in mutations
            )
        )

    scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"{assay_file} produced non-finite scores")
    return assay.with_columns(
        pl.Series("mutant", normalized_mutants),
        pl.Series(SCORE_COLUMN, scores),
    )


def parse_args() -> argparse.Namespace:
    """Parse RNA-ERNIE input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row_id", type=int, default=0)
    parser.add_argument(
        "--reference_sequences", type=Path, default=ConfigFitness.REFERENCE_FILE
    )
    parser.add_argument("--dms_directory", type=Path, default=ConfigFitness.ASSAY_DIR)
    parser.add_argument(
        "--output_directory",
        type=Path,
        default=ConfigFitness.PREDICTION_DIR / "RNAErnie",
    )
    parser.add_argument("--model_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path(__file__).with_name("src") / "vocab_1MER.txt",
    )
    return parser.parse_args()


def main() -> None:
    """Load RNA-ERNIE once and score the selected assay."""
    args = parse_args()
    reference = pl.read_csv(args.reference_sequences, encoding="utf8-lossy")
    if reference.columns and reference.columns[0].startswith("\ufeff"):
        first = reference.columns[0]
        reference = reference.rename({first: first.lstrip("\ufeff")})
    required_columns = {"DMS_ID", "RAW_CONSTRUCT_SEQ"}
    missing_columns = sorted(required_columns - set(reference.columns))
    if missing_columns:
        raise ValueError(
            f"{args.reference_sequences} is missing columns: {missing_columns}"
        )
    if args.row_id < 0 or args.row_id >= reference.height:
        raise IndexError(
            f"--row_id {args.row_id} is outside a {reference.height}-row reference"
        )

    row = reference.row(args.row_id, named=True)
    name = row["DMS_ID"]
    wild_type = normalize_sequence(row["RAW_CONSTRUCT_SEQ"])
    if len(wild_type) > 510:
        raise ValueError(f"{name} is too long for RNA-ERNIE: {len(wild_type)} nt")
    assay_file = args.dms_directory / f"{name}.csv"
    if not assay_file.is_file():
        raise FileNotFoundError(f"Fitness assay is missing: {assay_file}")
    output_file = args.output_directory / assay_file.name
    if output_file.exists():
        print(f"Skipping existing prediction: {output_file}")
        return

    model = ErnieForMaskedLM.from_pretrained(
        str(args.model_checkpoint), use_task_id=False
    )
    model.eval()
    converter = BatchConverter(vocab_path=args.vocab_path)
    result = score_assay(model, converter, assay_file, wild_type)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{name}.", dir=args.output_directory
    ) as directory:
        temporary_output = Path(directory) / output_file.name
        result.write_csv(temporary_output)
        temporary_output.replace(output_file)
    print(f"Scores saved to {output_file}")


if __name__ == "__main__":
    main()
