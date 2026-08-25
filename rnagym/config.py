"""Configuration shared across RNAGym benchmarks."""

import os
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_DIR / "data"
_DATABASE_DIR = Path(
    os.environ.get("RNAGYM_DATABASE_DIR", "/n/lw_groups/marks/databases")
)
_CHECKPOINT_DIR = Path(
    os.environ.get("RNAGYM_CHECKPOINT_DIR", "/n/lw_groups/marks/ckpt")
)


class _Config:
    """Configuration shared by every benchmark."""

    REPO_DIR = _REPO_DIR
    DATA_DIR = _DATA_DIR
    DATABASE_DIR = _DATABASE_DIR
    CHECKPOINT_DIR = _CHECKPOINT_DIR
    SEQUENCE_FILE = DATA_DIR / "rnagym_sequences.parquet"


class Config2D(_Config):
    """2D benchmark configuration."""

    DIR = _REPO_DIR / "rnagym" / "s2d"
    DATA_DIR = _DATA_DIR / "2d"
    RAW_DIR = DATA_DIR / "raw_data"
    MAPPING_FILE = DATA_DIR / "rnagym_mapping.parquet"
    STRUCTURE_FILE = DATA_DIR / "rnagym_2d.parquet"
    RFAM_FILE = DATA_DIR / "rnagym_rfams.parquet"
    PREDICTION_DIR = DATA_DIR / "predictions"
    LEADERBOARD_DIR = _REPO_DIR / "leaderboard" / "2d"
    LEADERBOARD_FILE = LEADERBOARD_DIR / "leaderboard.csv"
    LEADERBOARD_README = LEADERBOARD_DIR / "README.md"
    BPRNA_FILE = RAW_DIR / "bprna_1m_dbn.zip"
    EFOLD_LNCRNA_FILE = RAW_DIR / "lncRNA_nonFiltered.json"
    EFOLD_VIRAL_FILE = RAW_DIR / "viral_fragments.json"
    PSEUDOBASE_FILE = RAW_DIR / "pseudobase.csv"
    MC_ANNOTATE = DIR / ".pixi/model-sources/RNA_assessment/MC-Annotate"

    MIN_SEQUENCE_IDENTITY = 0.40
    MIN_COVERAGE = 0.80
    COVERAGE_MODE = 0
    CLUSTER_MODE = 0
    NUM_FOLDS = 5
    RANDOM_SEED = 42
    MMSEQS_THREADS = 1
    MAX_SEQUENCE_LENGTH = 1_022
    SCORE_BATCH_SIZE = 8192
    HEADLINE_MODIFIERS = ("1M7", "2A3", "DMS", "NMIA")
    TRAINING_OVERLAP = {
        "mapping": {"eternafold", "ribonanzanet"},
        "bprna": set(),
        "efold_challenging": set(),
        "pseudobase": set(),
        "pdb": {"ribonanzanet", "rna-fm", "ufold"},
    }
    RFAM_VERSION = "15.1"
    RFAM_DIR = _DATABASE_DIR / f"Rfam-{RFAM_VERSION}"
    RFAM_SHARDS = 16


class ConfigFitness(_Config):
    """Fitness benchmark configuration."""

    DIR = _REPO_DIR / "fitness"
    DATA_DIR = _DATA_DIR / "fitness"
    MSA_DIR = DATA_DIR / "msa"
    REFERENCE_FILE = DIR / "reference_sheet_final.csv"


class ConfigRiboseek(_Config):
    """Shared Riboseek configuration."""

    CACHE_DIR = _DATA_DIR / "riboseek" / "cache"
    RNACENTRAL_VERSION = "27.0"
    RNACENTRAL_DIR = _DATABASE_DIR / "RNAcentral" / RNACENTRAL_VERSION
    NT_VERSION = "2026-08-04"
    NT_DIR = _DATABASE_DIR / "NCBI-nt" / NT_VERSION
    DATABASE_RNACENTRAL = RNACENTRAL_DIR / "riboseek_gpu"
    DATABASE_NT = NT_DIR / "chunks" / "riboseek_gpu"
    MAX_TARGET_LENGTH = 10_000
    TARGET_OVERLAP = 2_000  # Maximum benchmark query length
    NT_PARTITION_DIR = NT_DIR / "partitions"
    DATABASE_NT_PARTS = tuple(
        sorted(
            path
            for path in NT_PARTITION_DIR.glob("riboseek_gpu_*")
            if path.is_file() and not path.suffix and not path.name.endswith("_h")
        )
    )


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
    CACHE_DIR = DATA_DIR / "cache"
    USALIGN_DIR = DATA_DIR / "usalign"
    USALIGN_ANNOTATION_DIR = USALIGN_DIR / "annotations"
    LEADERBOARD_DIR = _REPO_DIR / "leaderboard" / "3d"
    LEADERBOARD_FILE = LEADERBOARD_DIR / "leaderboard.csv"
    LEADERBOARD_README = LEADERBOARD_DIR / "README.md"
    RFAM_VERSION = "15.0"
    RFAM_DIR = _DATABASE_DIR / f"Rfam-{RFAM_VERSION}"
    AF3_DIR = DIR / ".pixi" / "model-sources" / "alphafold3"
    AF3_DATABASE_DIR = _DATABASE_DIR / "AlphaFold3" / "latest"
    AF3_PARAM_DIR = _CHECKPOINT_DIR / "alphafold3" / "latest"
    MC_ANNOTATE = DIR / ".pixi" / "model-sources" / "RNA_assessment" / "MC-Annotate"

    MAX_RESOLUTION = 5.0
    MAX_MISSING = 0.25
    MAX_UNKNOWN_RATIO = 0.10
    MAX_POLYMER_COVERAGE = 0.33
    MIN_LENGTH = 16
    MAX_LENGTH = 1024
    MAX_COMPLEX_LENGTH = 2000

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
            cls.MIN_LENGTH <= chain["L"] <= cls.MAX_LENGTH
            and chain["% covered (any polymer)"] <= cls.MAX_POLYMER_COVERAGE
            and chain["Self Structured"]
        )
