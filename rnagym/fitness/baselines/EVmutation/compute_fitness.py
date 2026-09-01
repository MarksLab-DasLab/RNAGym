"""Score fitness assays with EVmutation and their released Riboseek MSAs."""

import argparse
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from evcouplings.couplings import CouplingsModel
from evcouplings.mutate import extract_mutations
from rnagym.config import ConfigFitness

MUTATION_PATTERN = re.compile(r"^[ACGU]\d+[ACGU](?:,[ACGU]\d+[ACGU])*$")
SCORE_COLUMN = "prediction_epistatic"


def build_config(
    name: str,
    alignment_file: Path,
    query_file: Path,
    sequence_id: str,
    work_dir: Path,
    cpu_count: int,
) -> dict:
    """Build the EVcouplings existing-alignment pipeline configuration."""
    return {
        "stages": ["align", "couplings"],
        "pipeline": "protein_monomer",
        "global": {
            "alphabet": "rna",
            "prefix": str(work_dir / name),
            "theta": 0.9,
            "cpu": cpu_count,
            "region": None,
            "sequence_id": sequence_id,
            "sequence_file": str(query_file),
        },
        "align": {
            "alphabet": "rna",
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
            "alphabet": "rna",
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


def normalize_sequence(sequence: str) -> str:
    """Normalize an RNA sequence for reference and MSA consistency checks."""
    return sequence.upper().replace("T", "U").replace("-", "").replace(".", "")


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


def infer_model(config_file: Path, work_dir: Path) -> CouplingsModel:
    """Run EVcouplings and load the inferred model.

    EVcouplings may reject its downstream contact report when an alignment has
    no significant couplings. Mutation scoring only requires the Potts model,
    which PLMC has already saved at that point.
    """
    command = f"evcouplings_runcfg {shlex.quote(str(config_file))}"
    process = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
    )
    detail = process.stderr.strip() or process.stdout.strip()
    expected_bailout = "No couplings identified" in detail
    if process.returncode and not expected_bailout:
        raise RuntimeError(f"EVcouplings exited {process.returncode}: {detail}")
    model_files = list(work_dir.glob("**/*.model"))
    if len(model_files) != 1:
        raise RuntimeError(
            f"EVcouplings exited {process.returncode} and produced "
            f"{len(model_files)} models: {detail}"
        )
    return CouplingsModel(str(model_files[0]))


def score_assay(
    assay_file: Path,
    alignment_file: Path,
    query_file: Path,
    wild_type: str,
    name: str,
    work_dir: Path,
    cpu_count: int,
) -> pl.DataFrame:
    """Infer one Potts model and score every covered assay variant."""
    sequence_id, aligned_query = first_fasta_record(alignment_file)
    query_id, query_sequence = first_fasta_record(query_file)
    if sequence_id != query_id:
        raise ValueError(f"The Riboseek MSA and query identifiers differ for {name}")
    aligned_query = normalize_sequence(aligned_query)
    query_sequence = normalize_sequence(query_sequence)
    if aligned_query != wild_type or query_sequence != wild_type:
        raise ValueError(f"The Riboseek MSA query does not match {name}")

    config = build_config(
        name,
        alignment_file.resolve(),
        query_file.resolve(),
        sequence_id,
        work_dir,
        cpu_count,
    )
    config_file = work_dir / "config.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    model = infer_model(config_file, work_dir)
    covered_positions = set(model.index_list)

    assay = pl.read_csv(assay_file)
    required_columns = {"mutant", "sequence"}
    missing_columns = sorted(required_columns - set(assay.columns))
    if missing_columns:
        raise ValueError(f"{assay_file} is missing columns: {missing_columns}")

    scores = []
    for row, (mutant, sequence) in enumerate(
        assay.select("mutant", "sequence").iter_rows()
    ):
        mutations = parse_mutations(mutant, wild_type)
        variant = list(wild_type)
        for position, _, target in mutations:
            variant[position - 1] = target
        if sequence is None or normalize_sequence(sequence) != "".join(variant):
            raise ValueError(f"{assay_file} row {row} sequence disagrees with mutant")
        if any(position not in covered_positions for position, _, _ in mutations):
            scores.append(float("nan"))
            continue
        score = float(model.delta_hamiltonian(mutations)[0])
        if not np.isfinite(score):
            raise FloatingPointError(
                f"{assay_file} row {row} produced a nonfinite score"
            )
        scores.append(score)
    return assay.with_columns(pl.Series(SCORE_COLUMN, scores))


def parse_args() -> argparse.Namespace:
    """Parse EVmutation input and output locations."""
    parser = argparse.ArgumentParser(
        description="Score RNA fitness assays with EVmutation"
    )
    parser.add_argument(
        "--msa_dir", type=Path, default=ConfigFitness.MSA_DIR / "by_assay"
    )
    parser.add_argument("--ref_sheet", type=Path, default=ConfigFitness.REFERENCE_FILE)
    parser.add_argument("--dms_dir", type=Path, default=ConfigFitness.ASSAY_DIR)
    parser.add_argument(
        "--out_dir", type=Path, default=ConfigFitness.PREDICTION_DIR / "EVmutation"
    )
    parser.add_argument(
        "--tmp_dir", type=Path, default=ConfigFitness.DATA_DIR / "tmp" / "evmutation"
    )
    parser.add_argument("--cpu", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Score every reference-sheet assay that has a released Riboseek MSA."""
    args = parse_args()
    if args.cpu < 1:
        raise ValueError("--cpu must be positive")
    if not args.msa_dir.is_dir():
        raise FileNotFoundError(f"Riboseek MSA directory is missing: {args.msa_dir}")

    reference = pl.read_csv(args.ref_sheet, encoding="utf8-lossy")
    if reference.columns and reference.columns[0].startswith("\ufeff"):
        first = reference.columns[0]
        reference = reference.rename({first: first.lstrip("\ufeff")})
    required_columns = {"DMS_ID", "RAW_CONSTRUCT_SEQ"}
    missing_columns = sorted(required_columns - set(reference.columns))
    if missing_columns:
        raise ValueError(f"{args.ref_sheet} is missing columns: {missing_columns}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for name, raw_sequence in reference.select(
        "DMS_ID", "RAW_CONSTRUCT_SEQ"
    ).iter_rows():
        alignment_file = args.msa_dir / f"{name}.a3m"
        query_file = args.msa_dir / f"{name}.fa"
        if not alignment_file.is_file() and not query_file.is_file():
            continue
        if not alignment_file.is_file() or not query_file.is_file():
            raise FileNotFoundError(f"Incomplete Riboseek MSA files for {name}")
        assay_file = args.dms_dir / f"{name}.csv"
        if not assay_file.is_file():
            raise FileNotFoundError(f"Fitness assay is missing: {assay_file}")
        output_file = args.out_dir / assay_file.name
        if output_file.exists():
            continue
        if raw_sequence is None:
            raise ValueError(f"{name} has no reference sequence")

        print(f"Scoring {name}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=args.tmp_dir) as path:
            result = score_assay(
                assay_file,
                alignment_file,
                query_file,
                normalize_sequence(raw_sequence),
                name,
                Path(path),
                args.cpu,
            )
        with tempfile.TemporaryDirectory(
            prefix=f".{name}.{os.getpid()}-", dir=args.out_dir
        ) as output_staging_dir:
            temporary_output = Path(output_staging_dir) / output_file.name
            result.write_csv(temporary_output)
            temporary_output.replace(output_file)
        completed += 1
    print(f"Scored {completed} assay(s)")


if __name__ == "__main__":
    main()
