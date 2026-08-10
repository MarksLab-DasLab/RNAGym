"""Configuration shared across RNAGym benchmarks."""

import os
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_DIR / "data"
_DATABASE_DIR = Path(os.environ.get("RNAGYM_DATABASE_DIR", "/n/groups/marks/databases"))


class _Config:
    """Configuration shared by every benchmark."""

    REPO_DIR = _REPO_DIR
    DATABASE_DIR = _DATABASE_DIR


class Config2D(_Config):
    """2D benchmark configuration."""

    DIR = _REPO_DIR / "rnagym" / "s2d"
    DATA_DIR = _DATA_DIR / "2d"
    RAW_DIR = DATA_DIR / "raw_data"
    MAPPING_FILE = DATA_DIR / "rnagym_mapping.parquet"
    STRUCTURE_FILE = DATA_DIR / "rnagym_2d.parquet"
    SEQUENCE_FILE = DATA_DIR / "rnagym_sequences.parquet"
    RFAM_FILE = DATA_DIR / "rnagym_rfams.parquet"
    PREDICTION_DIR = DATA_DIR / "predictions"
    LEADERBOARD_DIR = _REPO_DIR / "leaderboard" / "2d"
    LEADERBOARD_FILE = LEADERBOARD_DIR / "leaderboard.csv"
    LEADERBOARD_README = LEADERBOARD_DIR / "README.md"
    PSEUDOBASE_FILE = RAW_DIR / "pseudobase.csv"
    MC_ANNOTATE = DIR / ".pixi/model-sources/RNA_assessment/MC-Annotate"

    MIN_SEQUENCE_IDENTITY = 0.40
    MIN_COVERAGE = 0.80
    COVERAGE_MODE = 0
    CLUSTER_MODE = 0
    NUM_FOLDS = 5
    RANDOM_SEED = 42
    MMSEQS_THREADS = 1
    SCORE_BATCH_SIZE = 8192
    HEADLINE_MODIFIERS = ("1M7", "2A3", "DMS", "NMIA")
    TRAINING_OVERLAP = {
        "mapping": {"eternafold", "ribonanzanet"},
        "pseudobase": set(),
        "pdb": {"ribonanzanet", "rna-fm", "ufold"},
    }
    RFAM_VERSION = "15.1"
    RFAM_DIR = _DATABASE_DIR / f"Rfam-{RFAM_VERSION}"
    RFAM_SHARDS = 16


class Config3D(_Config):
    """3D benchmark configuration."""

    DIR = _REPO_DIR / "rnagym" / "s3d"
    DATA_DIR = _DATA_DIR / "3d"
    CURATION_DIR = DATA_DIR / "curation"
    ANNOTATED_CHAINS_FILE = CURATION_DIR / "annotated_chains.csv"
    MERGED_PDB_IDS_FILE = CURATION_DIR / "merged_pdb_ids.csv"
    TARGET_FILE = DATA_DIR / "rnagym_3d.parquet"
    SCORE_FILE = DATA_DIR / "rnagym_3d_scores.parquet"
    STRUCTURE_DIR = DATA_DIR / "structures"
    MSA_DIR = DATA_DIR / "msa"
    PREDICTION_DIR = DATA_DIR / "predictions"
    OUT_DIR = DATA_DIR / "out"
    USALIGN_DIR = DATA_DIR / "usalign"
    LEADERBOARD_DIR = _REPO_DIR / "leaderboard" / "3d"
    LEADERBOARD_FILE = LEADERBOARD_DIR / "leaderboard.csv"
    LEADERBOARD_README = LEADERBOARD_DIR / "README.md"
    RFAM_VERSION = "15.0"
    RFAM_DIR = _DATABASE_DIR / f"Rfam-{RFAM_VERSION}"

    MAX_RESOLUTION = 5.0
    MAX_MISSING = 0.25
    MAX_POLYMER_COVERAGE = 0.33
    MIN_LENGTH = 16
    MAX_LENGTH = 2000

    @classmethod
    def passes_quality(cls, chain):
        """Apply the shared PDB download, resolution, and missing-data filters."""
        resolution = chain["Resolution"]
        return (
            chain["Asym. Chain ID"] != "ERROR: Failed to download"
            and isinstance(resolution, str)
            and resolution not in {".", "n.s."}
            and (
                resolution == "N/A"
                or max(map(float, resolution.split(","))) <= cls.MAX_RESOLUTION
            )
            and chain["Fraction missing"] <= cls.MAX_MISSING
        )

    @classmethod
    def is_monomer(cls, chain):
        """Apply the monomer filters to one annotated PDB chain."""
        if not cls.passes_quality(chain):
            return False
        return (
            cls.MIN_LENGTH <= chain["L"] <= cls.MAX_LENGTH
            and chain["% covered (any polymer)"] <= cls.MAX_POLYMER_COVERAGE
            and chain["Self Structured"]
        )
