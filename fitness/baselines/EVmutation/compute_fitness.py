#!/usr/bin/env python3
"""
Score non-coding RNA assays with EVmutation.
"""

import argparse
import re
import shlex
import shutil
import string
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from evcouplings.couplings import CouplingsModel
from evcouplings.mutate import predict_mutation_table


def run_command(command_str, output_handle=None):
    """Run shell command quietly using shlex."""
    try:
        cmd_parts = shlex.split(command_str)
        return subprocess.run(
            cmd_parts,
            check=True,
            capture_output=output_handle is None,
            stdout=output_handle,
            stderr=subprocess.PIPE if output_handle is not None else None,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"[ERROR] Command failed: {command_str}")
        print(f"{error.stderr.decode() if error.stderr else ''}")
        raise error


def get_rfam_match(sequence, name, database_path, temp_path, cpu_count=1, threshold=0.2):
    """
    Scan Rfam and return the accession with the largest alignment file with
    E-value less than the provided threshold.
    """
    if not re.search(r"ribozyme|trna|aptamer", name, re.IGNORECASE):
        return None

    query_fasta = temp_path / "query.fa"
    query_fasta.write_text(f">{name}\n{sequence}\n")
    table_out = temp_path / "rfam.tbl"

    try:
        run_command(
            f"cmscan --rfam --nohmmonly --cpu {cpu_count} --tblout {table_out} "
            f"{database_path}/Rfam.cm {query_fasta}"
        )
    except subprocess.CalledProcessError:
        return None

    if not table_out.exists():
        return None

    candidates = set()
    with open(table_out) as file_handle:
        for line in file_handle:
            if not line.startswith("#"):
                parts = line.split()
                if len(parts) > 15:
                    accession, e_value = parts[1], float(parts[15])
                    if e_value <= threshold:
                        candidates.add(parts[1])

    # Select deepest alignment based on file size
    best_accession = None
    max_file_size = -1
    for accession in candidates:
        full_path = database_path / "alignments" / f"{accession}.sto"
        current_size = full_path.stat().st_size
        if current_size > max_file_size:
            max_file_size = current_size
            best_accession = accession

    if best_accession is not None:
        print(f"  > Selected {best_accession} ({max_file_size / 1024**2:.1f} MB)")
    return best_accession


def prepare_alignment(accession, sequence, name, database_path, temp_path):
    """
    Fetch CM, align query, and merge with reference (full/seed) into consensus A2M.
    """
    # Select Reference (Full vs Seed logic)
    full_sto = database_path / "alignments" / f"{accession}.sto"

    # Threshold: 300 MB
    reference_file = (
        full_sto
        if full_sto.exists() and full_sto.stat().st_size <= 300 * 1024**2
        else None
    )

    if not reference_file:
        reference_file = temp_path / "ref_seed.sto"
        with open(reference_file, "w") as file_handle:
            run_command(
                f"esl-afetch {database_path}/Rfam.seed {accession}",
                output_handle=file_handle,
            )

    # Align Query
    cm_file = temp_path / f"{accession}.cm"
    with open(cm_file, "w") as file_handle:
        run_command(
            f"cmfetch {database_path}/Rfam.cm {accession}", output_handle=file_handle
        )

    query_sto = temp_path / "query.sto"
    run_command(f"cmalign --notrunc -o {query_sto} {cm_file} {temp_path}/query.fa")

    # Merge and filter (keeping uppercase and gaps, skipping insertions in
    # lowercase)
    trans_table = str.maketrans("", "", string.ascii_lowercase + ".")
    combined_a2m = temp_path / "combined.a2m"
    
    with open(combined_a2m, "w") as output_handle:
        for source_file in [query_sto, reference_file]:
            proc = run_command(
                f"esl-reformat a2m {source_file}",
            )
    
            for line in proc.stdout.splitlines(keepends=True):
                if line.startswith(">"):
                    output_handle.write(line)
                else:
                    output_handle.write(line.strip().translate(trans_table) + "\n")
    
    return combined_a2m


def run_evcouplings(
    job_name, aligned_a2m, dms_csv_path, sequence_id, temp_path, cpu_count
):
    """Configure, run pipeline, and score mutations."""
    config = {
        "stages": ["align", "couplings"],
        "pipeline": "protein_monomer",
        "global": {
            "alphabet": "rna",
            "prefix": str(temp_path / job_name),
            "theta": 0.9,
            "cpu": cpu_count,
            "region": None,
            "sequence_id": sequence_id,
            "sequence_file": str(temp_path / "query.fa"),
        },
        "align": {
            "alphabet": "rna",
            "protocol": "existing",
            "input_alignment": str(aligned_a2m),
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

    config_path = temp_path / "config.yaml"
    with open(config_path, "w") as file_handle:
        yaml.dump(config, file_handle)

    try:
        run_command(f"evcouplings_runcfg {config_path}")

        # Locate model file
        model_files = list(temp_path.glob("**/*.model"))
        if not model_files:
            print(f"[ERROR] Model file not found in {temp_path}")
            return None
        model_path = model_files[0]

        # Score
        couplings_model = CouplingsModel(str(model_path))
        dms_df = pd.read_csv(dms_csv_path)

        # Clean and Filter
        mutation_col = "mutant"
        dms_df["mutant_clean"] = (
            dms_df[mutation_col]
            .astype(str)
            .str.upper()
            .str.replace("T", "U")
            .str.replace(" ", "")
        )

        # Strict Regex
        valid_mask = dms_df["mutant_clean"].str.match(
            r"^([A-Z]\d+[A-Z])(,[A-Z]\d+[A-Z])*$"
        )
        valid_rows = dms_df[valid_mask]

        if len(valid_rows) < len(dms_df) - 1:
            raise ValueError("{dms_csv_path} contains invalid dms rows")

        # Score and save in output_file
        scoring_df = valid_rows.copy()
        scoring_df["mutant"] = scoring_df["mutant_clean"]
        predictions_df = predict_mutation_table(
            couplings_model, scoring_df, "prediction_epistatic"
        )
        score_map = predictions_df.set_index("mutant")["prediction_epistatic"]
        dms_df["prediction_epistatic"] = dms_df["mutant_clean"].map(score_map)
        output_file = temp_path / f"{job_name}_prediction_epistatic.csv"
        dms_df.drop(columns=["mutant_clean"]).to_csv(output_file, index=False)

        return output_file

    except Exception as error:
        print(f"[ERROR] EVcouplings/Scoring failed: {error}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Score RNA DMS assays using EVmutation"
    )
    parser.add_argument("--rfam_dir", type=Path, required=True)
    parser.add_argument("--ref_sheet", type=Path, required=True)
    parser.add_argument("--dms_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tmp_dir", type=Path, default=Path("./ev_tmp"))
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    args.tmp_dir.mkdir(exist_ok=True)

    rfam_cm_path = args.rfam_dir / "Rfam.cm"
    if not rfam_cm_path.exists():
        sys.exit(f"[ERROR] Rfam.cm missing in {args.rfam_dir}")

    reference_df = pd.read_csv(args.ref_sheet)
    for _, row in reference_df.iterrows():
        name = row["DMS_ID"]
        raw_seq = row["RAW_CONSTRUCT_SEQ"]
        sequence = raw_seq.upper().replace("T", "U").replace("N", "-")
        output_path = args.out_dir / f"{name}.csv"

        dms_path = args.dms_dir / f"{name}.csv"
        if output_path.exists() or not dms_path.exists():
            continue

        print(f"--- Processing {name} ---")
        assay_temp_dir = args.tmp_dir / name
        if assay_temp_dir.exists():
            shutil.rmtree(assay_temp_dir)
        assay_temp_dir.mkdir()

        try:
            rfam_id = get_rfam_match(
                sequence, name, args.rfam_dir, assay_temp_dir, args.cpu
            )

            if rfam_id:
                print(f"  > Aligning to {rfam_id}...")
                aligned_a2m = prepare_alignment(
                    rfam_id, sequence, name, args.rfam_dir, assay_temp_dir
                )

                print("  > Running Pipeline...")
                result_file = run_evcouplings(
                    name, aligned_a2m, dms_path, name, assay_temp_dir, args.cpu
                )

                if result_file:
                    shutil.copy(result_file, output_path)
                    print(f"  [SUCCESS] Saved {output_path}")
                else:
                    print("  [ERROR] Prediction failed.")
            else:
                print("  [SKIP] No Rfam match.")

        except Exception as error:
            print(f"  [ERROR] Exception processing {name}: {error}")
            raise error

        finally:
            if assay_temp_dir.exists():
                shutil.rmtree(assay_temp_dir)


if __name__ == "__main__":
    main()
