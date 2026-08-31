"""Configuration shared across RNAGym benchmarks."""

import os
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = Path(os.environ.get("RNAGYM_DATA_DIR", _REPO_DIR / "data"))
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

    DIR = _REPO_DIR / "rnagym" / "fitness"
    DATA_DIR = _DATA_DIR / "fitness"
    ASSAY_DIR = DATA_DIR / "assays"
    COMBINED_DIR = DATA_DIR / "merged"
    MSA_DIR = DATA_DIR / "msa"
    PREDICTION_DIR = DATA_DIR / "model_predictions"
    REFERENCE_FILE = DATA_DIR / "reference_sheet_final.csv"
    REPORT_DIR = DATA_DIR / "reports"


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
    # Preserve complete matches to any benchmark query across chunk boundaries
    TARGET_OVERLAP = 2_000
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
    CACHE_DIR = CURATION_DIR / ".cache"
    ANNOTATED_CHAINS_FILE = CURATION_DIR / "annotated_chains.parquet"
    TARGET_FILE = DATA_DIR / "rnagym_3d.parquet"
    SCORE_FILE = DATA_DIR / "rnagym_3d_scores.parquet"
    MSA_DIR = DATA_DIR / "msa"
    PREDICTION_DIR = DATA_DIR / "predictions"
    USALIGN_DIR = DATA_DIR / "usalign"
    USALIGN_ANNOTATION_DIR = USALIGN_DIR / "annotations"
    USALIGN_BINARY = "USalign"
    USALIGN_REFERENCES_FILE = USALIGN_DIR / "references.txt"
    LEADERBOARD_DIR = _REPO_DIR / "leaderboard" / "3d"
    LEADERBOARD_FILE = LEADERBOARD_DIR / "leaderboard.csv"
    LEADERBOARD_README = LEADERBOARD_DIR / "README.md"

    RNA3DB_RELEASE = "2026-01-05-full-release"
    RNA3DB_DIR = CACHE_DIR / "rna3db" / RNA3DB_RELEASE
    RNA3DB_JSON_DIR = RNA3DB_DIR / "jsons"
    RNA3DB_CMSCAN_DIR = RNA3DB_DIR / "cmscans"
    RNA3DB_PARSE_FILE = RNA3DB_JSON_DIR / "parse.json"
    RFAM_VERSION = "15.0"  # Matches the bundled RNA3DB annotations
    RFAM_DIR = _DATABASE_DIR / "Rfam" / RFAM_VERSION
    RFAM_CM = RFAM_DIR / "Rfam.cm"
    RFAM_FAMILY_TABLE = RFAM_DIR / "family.txt.gz"

    RCSB_ASSEMBLY_URL = (
        "https://files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/all/"
        "{pdb_id}-assembly1.cif.gz"
    )
    RCSB_FULL_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

    MODEL_SOURCE_DIR = DIR / ".pixi" / "model-sources"
    AF3_DIR = MODEL_SOURCE_DIR / "alphafold3"
    AF3_DATABASE_DIR = _DATABASE_DIR / "AlphaFold3" / "latest"
    AF3_PARAM_DIR = _CHECKPOINT_DIR / "alphafold3" / "latest"
    NUFOLD_DIR = MODEL_SOURCE_DIR / "nufold"
    NUFOLD_PARAM_FILE = _CHECKPOINT_DIR / "nufold" / "latest" / "global_step145245.pt"
    RHOFOLD_DIR = MODEL_SOURCE_DIR / "rhofold"
    RHOFOLD_PARAM_FILE = (
        _CHECKPOINT_DIR / "rhofold" / "latest" / "rhofold_pretrained_params.pt"
    )
    RF2NA_DIR = MODEL_SOURCE_DIR / "RoseTTAFold2NA"
    RF2NA_DATABASE = _DATABASE_DIR / "PDB100" / "latest" / "pdb100_2021Mar03"
    RF2NA_PARAM_FILE = _CHECKPOINT_DIR / "rosettafold2na" / "latest" / "RF2NA_apr23.pt"
    TRRNA_DIR = MODEL_SOURCE_DIR / "trRosettaRNA_v1.1"
    IPKNOT = MODEL_SOURCE_DIR / "ipknot" / "ipknot-1.1.0-x86_64-linux" / "ipknot"
    MC_ANNOTATE = MODEL_SOURCE_DIR / "RNA_assessment" / "MC-Annotate"

    COVERAGE_RADIUS = 4.5
    HYBRID_CUTOFF = 0.9
    MAX_COMPLEX_LENGTH = 2_000
    MAX_MODIFIED_RATIO = 0.25
    MAX_RESOLUTION = 5.0
    MAX_MISSING = 0.25
    MAX_UNKNOWN_RATIO = 0.10
    MAX_POLYMER_COVERAGE = 0.33
    MIN_ANNOTATION_LENGTH = 8
    MIN_LENGTH = 16
    MIN_STRUCTURED_CONTACTS = 6
    SELF_CONTACT_MIN_NEIGHBOR_DISTANCE = 6
    TARGET_CUTOFF = "2023-01-12"

    @classmethod
    def assembly_file(cls, pdb_id: str) -> Path:
        """Return the cached biological assembly for one PDB entry."""
        return cls.pdb_dir(pdb_id) / f"{pdb_id.lower()}-assembly.cif"

    @classmethod
    def chain_dir(cls, pdb_id: str, asym_id: str) -> Path:
        """Return the cache directory for one PDB chain."""
        return cls.CACHE_DIR / pdb_id.lower() / asym_id

    @classmethod
    def chain_file(cls, pdb_id: str, asym_id: str) -> Path:
        """Return the cached minimal PDB for one chain."""
        return cls.chain_dir(pdb_id, asym_id) / f"{asym_id}.pdb"

    @classmethod
    def full_file(cls, pdb_id: str) -> Path:
        """Return the cached asymmetric unit for one PDB entry."""
        return cls.pdb_dir(pdb_id) / f"{pdb_id.lower()}-full.cif"

    @classmethod
    def pdb_dir(cls, pdb_id: str) -> Path:
        """Return the cache directory for one PDB entry."""
        return cls.CACHE_DIR / pdb_id.lower()
