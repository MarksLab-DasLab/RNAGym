"""Score fitness assays with EVmutation and their released Riboseek MSAs."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from evcouplings.couplings import CouplingsModel
from evcouplings.mutate import extract_mutations, predict_mutation_table
from evcouplings.utils import BailoutException
from evcouplings.utils.pipeline import execute

from rnagym.config import ConfigFitness
from rnagym.fitness.data import read_reference, write_csv_atomically

MUTATION_PATTERN = re.compile(r"^[ACGU]\d+[ACGU](?:,[ACGU]\d+[ACGU])*$")
RNA_TO_DNA = str.maketrans("Uu", "Tt")
SCORE_COLUMN = "prediction_epistatic"


def build_config(
    name: str,
    alignment_file: Path,
    query_file: Path,
    sequence_id: str,
    work_dir: Path,
    cpu_count: int,
) -> dict[str, Any]:
    """Build the EVcouplings existing-alignment pipeline configuration."""
    return {
        "stages": ["align", "couplings"],
        "pipeline": "protein_monomer",
        "global": {
            "alphabet": "dna",
            "prefix": str(work_dir / name),
            "theta": 0.9,
            "cpu": cpu_count,
            "region": None,
            "sequence_id": sequence_id,
            "sequence_file": str(query_file),
        },
        "align": {
            "alphabet": "dna",
            "protocol": "existing",
            "input_alignment": str(alignment_file),
            "first_index": 1,
            "sequence_id": sequence_id,
            "seqid_filter": None,
            "focus_sequence": sequence_id,
            "compute_num_effective_seqs": True,
            "minimum_sequence_coverage": 50,
            "minimum_column_coverage": 70,
            "extract_annotation": True,
            "sequence_weights": "nogaps",
        },
        "couplings": {
            "protocol": "standard",
            "iterations": "100",
            "alphabet": "dna",
            "ignore_gaps": True,
            "lambda_J": 0.01,
            "lambda_J_times_Lq": True,
            "lambda_h": 0.01,
            "lambda_group": None,
            "scale_clusters": None,
            "reuse_ecs": False,
            "min_sequence_distance": 4,
            "scoring_model": "logistic_regression",
            "save_model": True,
        },
        "tools": {
            "jackhmmer": "jackhmmer",
            "plmc": "plmc",
            "hmmbuild": "hmmbuild",
            "hmmsearch": "hmmsearch",
            "hhfilter": "hhfilter",
            "psipred": "psipred",
            "cns": "cns",
            "maxcluster": "maxcluster64bit",
            "usalign": "USalign",
        },
        "databases": {
            "uniprot": "",
            "uniref100": "",
            "uniref90": "",
            "uniref50": "",
            "sequence_download_url": "",
            "sifts_mapping_table": "",
            "sifts_sequence_db": "",
        },
    }


def first_fasta_record(path: Path) -> tuple[str, str]:
    """Read the identifier and sequence from the first FASTA record."""
    identifier = None
    sequence = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    break
                identifier = line[1:].split(maxsplit=1)[0]
            elif identifier is None:
                raise ValueError(f"{path} has sequence data before its first header")
            else:
                sequence.append(line)
    if identifier is None or not sequence:
        raise ValueError(f"{path} has no FASTA records")
    return identifier, "".join(sequence)


def infer_model(config: dict[str, Any]) -> CouplingsModel:
    """Fit DNA couplings with the alignment's coverage mask and query coordinates."""
    alignment = execute(**(config | {"stages": ["align"]}))
    positions = prepare_dna_alignment(
        Path(alignment["alignment_file"]), alignment["focus_sequence"]
    )
    # Preserve upstream (L - 1)(q - 1) scaling after replacing masked bases with gaps
    couplings = config["couplings"] | {
        "lambda_J": config["couplings"]["lambda_J"] * (len(positions) - 1) * 3,
        "lambda_J_times_Lq": False,
    }
    output = execute(**(config | {"stages": ["couplings"], "couplings": couplings}))
    model = CouplingsModel(output["model_file"])
    if not np.array_equal(model.index_list, positions):
        raise ValueError("PLMC positions disagree with the alignment coverage mask")
    return model


def main() -> None:
    """Score every reference-sheet assay that has a released Riboseek MSA."""
    args = parse_args()
    msa_dir = ConfigFitness.MSA_DIR / "by_assay"
    work_dir = ConfigFitness.DATA_DIR / "tmp/evmutation"
    if args.cpu < 1:
        raise ValueError("--cpu must be positive")
    if not msa_dir.is_dir():
        raise FileNotFoundError(f"Riboseek MSA directory is missing: {msa_dir}")

    reference = read_reference(ConfigFitness.REFERENCE_FILE)
    args.output.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    rejected = 0
    for name, raw_sequence in reference.select(
        "DMS_ID", "RAW_CONSTRUCT_SEQ"
    ).iter_rows():
        alignment_file = msa_dir / f"{name}.a3m"
        query_file = msa_dir / f"{name}.fa"
        if not alignment_file.is_file() and not query_file.is_file():
            continue
        if not alignment_file.is_file() or not query_file.is_file():
            raise FileNotFoundError(f"Incomplete Riboseek MSA files for {name}")
        assay_file = ConfigFitness.ASSAY_DIR / f"{name}.csv"
        if not assay_file.is_file():
            raise FileNotFoundError(f"Fitness assay is missing: {assay_file}")
        output_file = args.output / assay_file.name
        if output_file.exists():
            continue
        if raw_sequence is None:
            raise ValueError(f"{name} has no reference sequence")

        print(f"Scoring {name}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=work_dir) as path:
                result = score_assay(
                    assay_file,
                    alignment_file,
                    query_file,
                    normalize_sequence(raw_sequence),
                    name,
                    Path(path),
                    args.cpu,
                )
        except BailoutException as error:
            print(f"Skipped {name}: {error}", flush=True)
            rejected += 1
            continue
        write_csv_atomically(result, output_file)
        completed += 1
    print(f"Scored {completed} assay(s), EVcouplings rejected {rejected}")


def normalize_sequence(sequence: str) -> str:
    """Normalize an RNA sequence for reference and MSA consistency checks."""
    return sequence.upper().replace("T", "U").replace("-", "").replace(".", "")


def parse_args() -> argparse.Namespace:
    """Parse EVmutation input and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ConfigFitness.PREDICTION_DIR / "EVmutation"
    )
    parser.add_argument("--cpu", type=int, default=4)
    return parser.parse_args()


def parse_mutations(mutant: str | None, wild_type: str) -> list[tuple[int, str, str]]:
    """Parse one variant and verify its reference bases and positions."""
    if mutant is None or not mutant.strip() or mutant.strip().lower() in {"wt", "wild"}:
        return []
    cleaned = mutant.upper().replace("T", "U").replace(" ", "")
    if MUTATION_PATTERN.fullmatch(cleaned) is None:
        raise ValueError(f"Invalid mutation string: {mutant!r}")
    mutations = extract_mutations(cleaned)
    positions = [position for position, _, _ in mutations]
    if len(positions) != len(set(positions)):
        raise ValueError(f"Mutation string repeats a position: {mutant!r}")
    for position, source, _ in mutations:
        if position < 1 or position > len(wild_type):
            raise ValueError(
                f"Mutation position {position} is outside a {len(wild_type)} nt construct"
            )
        if wild_type[position - 1] != source:
            raise ValueError(
                f"Mutation {mutant!r} expects {source} at position {position}, "
                f"but the reference has {wild_type[position - 1]}"
            )
    return mutations


def prepare_dna_alignment(path: Path, focus_id: str) -> list[int]:
    """Convert protein A2M gaps and focus masking to PLMC's DNA representation."""
    # PLMC handles dot gaps and lowercase focus masking only for its protein alphabet
    # https://github.com/debbiemarkslab/plmc/blob/9152c483e18263a90cbe79082a61e83cf6dd70a1/src/plm.c#L248-L258
    # https://github.com/debbiemarkslab/plmc/blob/9152c483e18263a90cbe79082a61e83cf6dd70a1/src/plm.c#L395-L413
    destination = path.with_suffix(".dna.a2m")
    focus = False
    query = []
    with path.open() as source, destination.open("w") as output:
        for line in source:
            if line.startswith(">"):
                focus = line[1:].split()[0] == focus_id
                output.write(line)
            else:
                if focus:
                    query.extend(line.strip())
                    # Gap-reduced PLMC excludes query gaps while retaining original indices
                    line = "".join("-" if base.islower() else base for base in line)
                output.write(line.upper().replace(".", "-"))
    positions = [i + 1 for i, base in enumerate(query) if base in "ACGT"]
    if not positions:
        raise ValueError("Alignment has no covered query positions")
    destination.replace(path)
    return positions


def score_assay(
    assay_file: Path,
    alignment_file: Path,
    query_file: Path,
    wild_type: str,
    name: str,
    work_dir: Path,
    cpu_count: int,
) -> pl.DataFrame:
    """Score covered variants, retaining nulls for mutations outside modeled sites."""
    sequence_id, aligned_query = first_fasta_record(alignment_file)
    query_id, query_sequence = first_fasta_record(query_file)
    if sequence_id != query_id:
        raise ValueError(f"The Riboseek MSA and query identifiers differ for {name}")
    aligned_query = normalize_sequence(aligned_query)
    query_sequence = normalize_sequence(query_sequence)
    if aligned_query != wild_type or query_sequence != wild_type:
        raise ValueError(f"The Riboseek MSA query does not match {name}")

    dna_alignment_file = work_dir / "alignment.a3m"
    dna_query_file = work_dir / "query.fa"
    write_dna_fasta(alignment_file, dna_alignment_file)
    write_dna_fasta(query_file, dna_query_file)
    config = build_config(
        name,
        dna_alignment_file,
        dna_query_file,
        sequence_id,
        work_dir,
        cpu_count,
    )
    model = infer_model(config)

    assay = pl.read_csv(assay_file)
    required_columns = {"mutant", "sequence"}
    missing_columns = sorted(required_columns - set(assay.columns))
    if missing_columns:
        raise ValueError(f"{assay_file} is missing columns: {missing_columns}")

    normalized_mutants = []
    covered = []
    for row, (mutant, sequence) in enumerate(
        assay.select("mutant", "sequence").iter_rows()
    ):
        mutations = parse_mutations(mutant, wild_type)
        variant = list(wild_type)
        for position, _, target in mutations:
            variant[position - 1] = target
        if sequence is None or normalize_sequence(sequence) != "".join(variant):
            raise ValueError(f"{assay_file} row {row} sequence disagrees with mutant")
        normalized_mutants.append(
            (
                ",".join(
                    f"{source}{position}{target}"
                    for position, source, target in mutations
                )
                or "wt"
            ).translate(RNA_TO_DNA)
        )
        covered.append(all(position in model.index_map for position, _, _ in mutations))

    # predict_mutation_table is the upstream scoring API and requires pandas
    prediction_input = pd.DataFrame(
        {
            "mutant": [
                mutant for mutant, keep in zip(normalized_mutants, covered) if keep
            ]
        }
    )
    if prediction_input.empty:
        raise BailoutException("No assay variants fall within modeled positions")
    predictions = predict_mutation_table(model, prediction_input, SCORE_COLUMN)
    values = predictions[SCORE_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        count = int((~np.isfinite(values)).sum())
        raise FloatingPointError(f"{assay_file} produced {count} non-finite scores")
    scores = np.full(assay.height, np.nan)
    scores[covered] = values
    print(f"{name}: scored {sum(covered)}/{assay.height} variants within modeled sites")
    return assay.with_columns(pl.Series(SCORE_COLUMN, scores, nan_to_null=True))


def write_dna_fasta(source: Path, destination: Path) -> None:
    """Write a FASTA copy with uracil represented as thymine."""
    # Existing-alignment parsing uses the protein alphabet, which omits U
    # https://github.com/debbiemarkslab/EVcouplings/blob/14c83457c6cfca8156aabe0615067447a2169791/evcouplings/align/protocol.py#L702
    with source.open() as input_handle, destination.open("w") as output_handle:
        for line in input_handle:
            output_handle.write(
                line if line.startswith(">") else line.translate(RNA_TO_DNA)
            )


if __name__ == "__main__":
    main()
