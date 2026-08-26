"""Configuration shared across RNAGym benchmarks."""

import os
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_DIR / "data"
_DATABASE_DIR = Path(
    os.environ.get("RNAGYM_DATABASE_DIR", "/n/lw_groups/marks/databases")
)


class _Config:
    """Configuration shared by every benchmark."""

    REPO_DIR = _REPO_DIR
    DATA_DIR = _DATA_DIR
    DATABASE_DIR = _DATABASE_DIR
    MAX_SEQUENCE_LENGTH = 1_022
    RFAM_FILE = DATA_DIR / "rnagym_rfams.parquet"
    SEQUENCE_FILE = DATA_DIR / "rnagym_sequences.parquet"


class Config2D(_Config):
    """2D benchmark configuration."""

    DIR = _REPO_DIR / "rnagym" / "s2d"
    DATA_DIR = _DATA_DIR / "2d"
    RAW_DIR = DATA_DIR / "raw_data"
    MAPPING_FILE = DATA_DIR / "rnagym_mapping.parquet"
    STRUCTURE_FILE = DATA_DIR / "rnagym_2d.parquet"
    PREDICTION_DIR = DATA_DIR / "predictions"
    LEADERBOARD_DIR = _REPO_DIR / "leaderboard" / "2d"
    LEADERBOARD_FILE = LEADERBOARD_DIR / "leaderboard.csv"
    LEADERBOARD_README = LEADERBOARD_DIR / "README.md"
    BPRNA_FILE = RAW_DIR / "bprna_1m_dbn.zip"
    EFOLD_LNCRNA_FILE = RAW_DIR / "lncRNA_nonFiltered.json"
    EFOLD_VIRAL_FILE = RAW_DIR / "viral_fragments.json"
    PSEUDOBASE_FILE = RAW_DIR / "pseudobase.csv"
    MODEL_SOURCE_DIR = DIR / ".pixi" / "model-sources"
    MODEL_WEIGHT_DIR = DIR / ".pixi" / "model-weights"
    MC_ANNOTATE = MODEL_SOURCE_DIR / "RNA_assessment" / "MC-Annotate"

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
        "bprna": {"mxfold2", "ribonanzanet", "rinalmo", "rna-fm", "ufold"},
        "efold_challenging": set(),
        "pseudobase": set(),
        "pdb": {"ribonanzanet", "rinalmo", "rna-fm", "ufold"},
    }
    RFAM_VERSION = "15.1"
    RFAM_DIR = _DATABASE_DIR / "Rfam" / RFAM_VERSION
    RFAM_SHARDS = 16


class ConfigFitness(_Config):
    """Fitness benchmark configuration."""

    DIR = _REPO_DIR / "fitness"
    REFERENCE_FILE = DIR / "reference_sheet_final.csv"


class Config3D(_Config):
    """3D inputs shared with the 2D benchmark."""

    DATA_DIR = _DATA_DIR / "3d"
    CURATION_DIR = DATA_DIR / "curation"
    ANNOTATED_CHAINS_FILE = CURATION_DIR / "annotated_chains.csv"
    TARGET_FILE = DATA_DIR / "rnagym_3d.parquet"
    CACHE_DIR = DATA_DIR / "cache"

    MAX_RESOLUTION = 5.0
    MAX_MISSING = 0.25
    MAX_UNKNOWN_RATIO = 0.10
    MAX_POLYMER_COVERAGE = 0.33
    MIN_LENGTH = 16

    @classmethod
    def passes_quality(cls, chain):
        """Apply the shared PDB quality filters."""
        resolution = chain["Resolution"]
        sequence = chain["Sequence (unmod.)"]
        if not isinstance(resolution, str) or resolution in {".", "n.s."}:
            return False
        if resolution != "N/A":
            resolutions = [
                float(value) for value in resolution.split(",") if value.strip() != "?"
            ]
            if not resolutions or max(resolutions) > cls.MAX_RESOLUTION:
                return False
        return (
            chain["Asym. Chain ID"] != "ERROR: Failed to download"
            and chain["Fraction missing"] <= cls.MAX_MISSING
            and sequence.count("N") / len(sequence) <= cls.MAX_UNKNOWN_RATIO
        )

    @classmethod
    def is_monomer(cls, chain):
        """Apply the monomer filters to one annotated PDB chain."""
        if not cls.passes_quality(chain):
            return False
        return (
            cls.MIN_LENGTH <= chain["L"] <= cls.MAX_SEQUENCE_LENGTH
            and chain["% covered (any polymer)"] <= cls.MAX_POLYMER_COVERAGE
            and chain["Self Structured"]
        )
