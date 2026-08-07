"""Reproduce the published RNAGym fitness benchmark results."""

import io
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import requests

BASE_URL = "https://marks.hms.harvard.edu/rnagym/fitness_prediction"
FITNESS_DIR = Path(__file__).resolve().parents[1] / "fitness"

RESULT_FILES = (
    "results_by_rna_type.csv",
    "results_by_mutation_depth.csv",
    "results_by_rna_type_and_depth.csv",
    "assay_level_results.csv",
    "assay_level_results_transposed.csv",
)


def download(url: str, destination: Path) -> None:
    """Download and extract a ZIP archive."""
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    zipfile.ZipFile(io.BytesIO(response.content)).extractall(destination)


def run(command: str) -> None:
    """Run a command."""
    subprocess.run(shlex.split(command), check=True)


@pytest.fixture(scope="module")
def fitness_results(tmp_path_factory):
    """Download data and reproduce the benchmark results."""
    tmpdir = tmp_path_factory.mktemp("fitness")
    for archive in ("fitness_processed_assays", "model_predictions", "fitness_results"):
        destination = tmpdir / archive
        destination.mkdir()
        download(f"{BASE_URL}/{archive}.zip", destination)

    merged_dir = tmpdir / "merged"
    run(
        f"{sys.executable} {FITNESS_DIR}/merge_scoring_files.py "
        f"--processed_folder {tmpdir}/fitness_processed_assays/assays "
        f"--model_predictions_folder {tmpdir}/model_predictions/model_predictions "
        f"--output_folder {merged_dir}"
    )
    performance_dir = tmpdir / "performance"
    run(
        f"{sys.executable} {FITNESS_DIR}/performance_fitness.py "
        f"--reference_file {FITNESS_DIR}/reference_sheet_final.csv "
        f"--combined_dir {merged_dir} --performance_dir {performance_dir} --type all"
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
    pd.testing.assert_frame_equal(pd.read_csv(expected), pd.read_csv(actual))
