"""
Reproducibility test for the RNAGym fitness benchmark.

Downloads published data and results from marks.hms.harvard.edu/rnagym/,
re-runs the pipeline, and diffs output against the published results.
"""

import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import requests

BASE_URL = "https://marks.hms.harvard.edu/rnagym/fitness_prediction"
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "fitness"

RESULT_FILES = [
    "results_by_rna_type.csv",
    "results_by_mutation_depth.csv",
    "results_by_rna_type_and_depth.csv",
    "assay_level_results.csv",
    "assay_level_results_transposed.csv",
]


def download_and_unzip(url: str, dest: Path) -> None:
    """Download a zip file and extract it to dest."""
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    zip_path = dest / "archive.zip"
    zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()


@pytest.fixture(scope="module")
def fitness_results(tmp_path_factory):
    """Download data, run the pipeline, and return (expected_dir, actual_dir)."""
    tmpdir = tmp_path_factory.mktemp("fitness")

    # Download published data
    for archive in ("fitness_processed_assays", "model_predictions", "fitness_results"):
        dest = tmpdir / archive
        dest.mkdir()
        download_and_unzip(f"{BASE_URL}/{archive}.zip", dest)

    # Run merge_scoring_files.py
    merged_dir = tmpdir / "merged"
    subprocess.run(
        [
            "python",
            str(SCRIPT_DIR / "merge_scoring_files.py"),
            "--processed_folder", str(tmpdir / "fitness_processed_assays" / "assays"),
            "--model_predictions_folder", str(tmpdir / "model_predictions" / "model_predictions"),
            "--output_folder", str(merged_dir),
        ],
        check=True,
    )

    # Run performance_fitness.py
    performance_dir = tmpdir / "performance"
    subprocess.run(
        [
            "python",
            str(SCRIPT_DIR / "performance_fitness.py"),
            "--reference_file", str(SCRIPT_DIR / "reference_sheet_final.csv"),
            "--combined_dir", str(merged_dir),
            "--performance_dir", str(performance_dir),
            "--type", "all",
        ],
        check=True,
    )

    return tmpdir / "fitness_results", performance_dir


@pytest.mark.parametrize("csv_name", RESULT_FILES)
def test_result_matches_published(fitness_results, csv_name):
    expected_dir, actual_dir = fitness_results
    expected = expected_dir / csv_name
    actual = actual_dir / csv_name

    if not expected.exists():
        pytest.skip(f"{csv_name} not in published results")

    assert actual.exists(), f"{csv_name} was not produced by the pipeline"

    df_expected = pd.read_csv(expected)
    df_actual = pd.read_csv(actual)

    pd.testing.assert_frame_equal(df_expected, df_actual)
