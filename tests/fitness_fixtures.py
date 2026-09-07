"""Real assay inputs shared by the fast and native fitness checks."""

import shutil
from pathlib import Path

import pytest

from rnagym.config import ConfigFitness

REPOSITORY = Path(__file__).parent.parent
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "fitness"
ASSAY_DIR = FIXTURE_DIR / "assays"
REFERENCE_FILE = FIXTURE_DIR / "reference.csv"
PREDICTION_DIR = FIXTURE_DIR / "predictions" / "rna_fm_4fill"
ASSAY_NAMES = (
    "Andreasson_2020_ribozyme.csv",
    "Domingo_2018_tRNA.csv",
    "Tome_2014_GFP_aptamer.csv",
)


@pytest.fixture
def fitness_data(tmp_path, monkeypatch):
    """Install mutable copies of the released fixtures in an isolated data directory."""
    root = tmp_path / "fitness"
    root.mkdir()
    shutil.copy2(REFERENCE_FILE, root / "reference_sheet_final.csv")
    shutil.copytree(ASSAY_DIR, root / "assays")
    shutil.copytree(PREDICTION_DIR.parent, root / "model_predictions")
    shutil.copytree(FIXTURE_DIR / "msa", root / "msa/by_assay")
    monkeypatch.setenv("RNAGYM_DATA_DIR", str(tmp_path))
    for name, path in {
        "DATA_DIR": root,
        "ASSAY_DIR": root / "assays",
        "REFERENCE_FILE": root / "reference_sheet_final.csv",
        "PREDICTION_DIR": root / "model_predictions",
        "MSA_DIR": root / "msa",
        "COMBINED_DIR": root / "merged",
        "REPORT_DIR": root / "reports",
    }.items():
        monkeypatch.setattr(ConfigFitness, name, path)
    return root
