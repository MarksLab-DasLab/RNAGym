#!/usr/bin/env python3

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from rna3db.tabular import TabularOutput, read_tbls_from_dir

###############################################################################
# `config.py`:  Configuration variables for RNA Gym 3D.
###############################################################################
# ---- General ---- #
OUT_DIR = "./out"
PDB_OUT_PREFIX = OUT_DIR + "/{pdb_id}"
CHAIN_OUT_PREFIX = PDB_OUT_PREFIX + "/{chain_id}"
CHAIN_MINIMAL_PDB_FILE = CHAIN_OUT_PREFIX + "/{chain_id}.pdb"


def get_out_prefix(pdb_id: str, chain_id: str = "") -> Path:
    """
    Gets the output prefix for a given PDB ID and chain.
    """
    return Path(CHAIN_OUT_PREFIX.format(pdb_id=pdb_id, chain_id=chain_id)).resolve()


def get_minimal_pdb_file(pdb_id: str, chain_id: str) -> Path:
    """
    Gets the output prefix for a given PDB ID and chain.
    """
    return Path(
        CHAIN_MINIMAL_PDB_FILE.format(pdb_id=pdb_id, chain_id=chain_id)
    ).resolve()


# ---- Annotation ---- #
RNA_MIN_NT = 8  # Minimum length of RNA to annotate
# Maximum length of RNA to annotate.  Should be infinite to ensure proper
# TM_train calculations.
RNA_MAX_NT = float("inf")
MAX_RFAM_MSAS = float("inf")  # Don't search Rfam in structures with > this many
HYBRID_CUTOFF = 0.9  # Chains >CUTOFF% residue type will not be marked as hybrid
MIN_STRUCTURED_CONTACTS = 6  # This many self-contacting residues => structured
SELF_CONTACT_MIN_NEIGHBOR_DIST = 6  # Min. residue separation for self-contacts
COVERAGE_RADIUS = 4.5  # Radius used to determine polymer coverage

# Protein/RNA family HMM/CM databases
RFAM_DIR = "/path/to/Rfam-15.0"
RFAM_CM = f"{RFAM_DIR}/Rfam.cm"
RFAM_SEED = f"{RFAM_DIR}/Rfam.utf8.seed"
RFAM_FULL_ALIGNMENT = f"{RFAM_DIR}/alignments/{{accession}}.sto"
RFAM_ALIGNMENT_STATS = f"{RFAM_DIR}/alignments/{{accession}}.stat"
RNA3DB_CMSCANS_DIR = "./datasets/rna3db/cmscans"
RNA3DBENCH_DIR = "./datasets/2024_3d_bench"
RNA3DBENCH_DS3 = RNA3DBENCH_DIR + "/Dataset3/Dataset3_new.csv"
RNA3DBENCH_DS4 = RNA3DBENCH_DIR + "/Dataset4/Dataset4.csv"
RNA3DBENCH_DS3_CHAINS = {
    f"{pdb_id.lower()}_{auth_id}"
    for _, pdb_id, auth_id in pd.read_csv(RNA3DBENCH_DS3)[
        ["PDB ID", "Chain ID"]
    ].itertuples()
}
RNA3DBENCH_DS4_CHAINS = {
    f"{pdb_id.lower()}_{auth_id}"
    for _, pdb_id, auth_id in pd.read_csv(RNA3DBENCH_DS4)[
        ["PDB ID", "Chain ID"]
    ].itertuples()
}

# External links
# NOTE(MCA): In the vast majority of PDBs, biological assembly 1 is the unit
#  of interest for structure prediction.  In a few, biological assembly 2 is
#  also interesting.  Note that in some cases the biological assembly may be
#  "split" across the unit cell.
RCSB_ASSEMBLY_URL = "https://files.wwpdb.org/pub/pdb/data/assemblies/mmCIF/all/{pdb_id}-assembly1.cif.gz"
RCSB_ASSEMBLY_FILE = PDB_OUT_PREFIX + "/{pdb_id}-assembly.cif"
RCSB_FULL_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
RCSB_FULL_FILE = PDB_OUT_PREFIX + "/{pdb_id}-full.cif"

# --- Splitting ---
MAX_RESOLUTION = 5.0
RFAM_GOOD_CUTOFF = 1.00  # max E-value to be considered "good" Rfam hit
RFAM_BAD_CUTOFF = 1.00  # min. E-value to be considered "bad" Rfam hit
MAX_FRAC_MISSING = 0.25  # Max fraction of missing residues to consider
MIN_RFAM_OBSERVED = 0.00  # Min. fraction of the Rfam hit observed in the chain
# ALLOWABLE_COFACTORS = {"BA", "CA", "CL", "F", "FE", "K", "MG", "NI", "PO4", "SO4", "ZN"}
MAX_CLUSTER_E_VALUE = 1.0
MAX_PCT_COVER_MONOMER = 0.33
TOP_N = 3  # The top # of sequence clusters to select from each Rfam
# Minimum length should be double the annotation limit to ensure proper
# TM_train calculations.
MIN_L = RNA_MIN_NT * 2
MAX_N = 2000  # Multimers up to this many AA/NA residues will be considered
MONOMER_CSV = Path("monomer.csv").resolve()
MULTIMER_CSV = Path("complex.csv").resolve()
MONOMER_ANALYZED_CSV = Path("out/monomer.analyzed.csv").resolve()
MULTIMER_ANALYZED_CSV = Path("out/multimer.analyzed.csv").resolve()

# --- RNA3DB Sequence/Structure Clusters ---
RNA3DB_DIR = "./datasets/rna3db"
SEQ_CLUST_FILE = RNA3DB_DIR + "/cluster.seq.json"
STRUCT_CLUST_FILE = RNA3DB_DIR + "/cluster.struct.json"
with open(SEQ_CLUST_FILE) as f:
    SEQ_CLUST = json.load(f)
with open(STRUCT_CLUST_FILE) as f:
    STRUCT_CLUST = json.load(f)

# Mapping from homologous chain IDs (<pdb_id>_<auth_id>) to the representative
# parent chain ID of the cluster
# NOTE(MCA): Each representative chain is represented within its own cluster.
SEQ_CLUST_REPR_CHAINS = {
    homo_chain_id: repr_chain_id
    for repr_chain_id, seq_cluster in SEQ_CLUST.items()
    for homo_chain_id in seq_cluster.keys()
}

# Full table of all hits
RNA3DB_TABLE = read_tbls_from_dir(RNA3DB_CMSCANS_DIR)

# Mapping from <pdb_id>_<auth_id> -> hit
RNA3DB_HITS = {}
for hit in RNA3DB_TABLE:
    if hit.query_name not in RNA3DB_HITS:
        RNA3DB_HITS[hit.query_name] = TabularOutput(hits=[hit])
    else:
        RNA3DB_HITS[hit.query_name].hits.append(hit)

# Mapping from chain IDs to their the cluster component index (which cluster
# the chain belongs to)
# NOTE(MCA): Component IDs come in the form "component_<#>"
STRUCT_CLUST_COMPONENTS = {
    chain_id: int(component_id.split("_")[1])
    for component_id, chains in STRUCT_CLUST.items()
    for chain_id in chains.keys()
}

# Mapping from component ID to Rfam names and vice-versa
COMPONENT_RFAMS = {}  # Map from component ID to Rfam names in the cluster
RFAM_COMPONENTS = {}  # Map from Rfam name to component ID
COMPONENT_RFAMS[0] = set()  # No Rfam hits in component 0 by definition
for comp_id, chains in STRUCT_CLUST.items():
    comp_id = int(comp_id.split("_")[1])
    for chain_id in chains.keys():
        for hit in RNA3DB_HITS.get(chain_id, []):
            if hit.e_value <= MAX_CLUSTER_E_VALUE:
                COMPONENT_RFAMS.setdefault(comp_id, set()).add(hit.target_name)
                RFAM_COMPONENTS[hit.target_name] = comp_id

# ---- Analysis ---- #
SEQUENCE_ID = "{pdb_id}_{chain_id}"
RNA3DBENCH_INPUT_DIR = RNA3DBENCH_DIR + "/Dataset3/inputs/{sequence_id}"
CHAINS_DIR = OUT_DIR + "/chains"  # directory containing all RNA chain PDBs
CHAINS_LIST_FILE = CHAINS_DIR + "/chains.txt"  # list of all RNA chains to align to
# General EVCouplings configurations (replace with paths if not in $PATH)
FONT = "Nimbus Sans"  # plot font
TOOLS = {
    "jackhmmer": "jackhmmer",
    "plmc": "plmc",
    "hmmbuild": "hmmbuild",
    "hmmsearch": "hmmsearch",
    "hhfilter": "hhfilter",
    "psipred": "psipred",
    "cns": "cns",
    "maxcluster": "maxcluster64bit",
    "usalign": "USalign",
}
DATABASES = {
    "uniprot": "/path/to/uniprot/uniprot.fasta",
    "uniref100": "/path/to/uniref100/uniref100.fasta",
    "uniref90": "/path/to/uniref90/uniref90.fasta",
    "uniref50": "/path/to/uniref50/uniref50.fasta",
    "sequence_download_url": "http://rest.uniprot.org/uniprot/{}.fasta",
    "sifts_mapping_table": "/path/to/SIFTS/pdb_chain_uniprot_plus.csv",
    "sifts_sequence_db": "/path/to/SIFTS/pdb_chain_uniprot_plus.fasta",
}


# --- Evaluation ---
@dataclass(frozen=True)
class Baseline:
    """
    Stores information needed by RNAGym for a given baseline.
    """

    name: str
    install_dir: Path
    out_dir: Path
    job_sh: Path
    training_cutoff: str  # Date as YYYY-MM-DD

    # Output files relative to output_dir
    out_pdb: Path
    afa_file: Path
    mul_afa_file: Optional[Path]


PREDICTIONS_DIR = Path("./predictions").resolve()
AF3 = Baseline(
    name="af3",
    install_dir=Path("/path/to/alphafold3"),
    out_dir=PREDICTIONS_DIR / "af3",
    job_sh=Path("./scripts/af3.sh").resolve(),
    out_pdb="{name_lower}/{name_lower}_model.cif",
    afa_file="sequence.a3m",
    mul_afa_file="sequence_{asym_id}.a3m",
    training_cutoff="2023-01-12",  # doi.org/10.1038/s41586-024-07487-w (Data Availability)
)

NUFOLD = Baseline(
    name="nu",
    install_dir=Path("/path/to/nufold"),
    out_dir=PREDICTIONS_DIR / "nu",
    job_sh=Path("./scripts/nufold.sh").resolve(),
    out_pdb="output/{name_upper}/{name_upper}_rank_1.pdb",
    afa_file="input/{chain_key_upper}/{chain_key_upper}.a3m",
    mul_afa_file=None,
    training_cutoff="2022-02-28",  # doi.org/10.1038/s41467-025-56261-7 (Results)
)

RF2NA = Baseline(
    name="rf2na",
    install_dir=Path("/path/to/RoseTTAFold2NA"),
    out_dir=PREDICTIONS_DIR / "rf2na",
    job_sh=Path("./scripts/rf2na.sh").resolve(),
    out_pdb="models/model_00.pdb",
    afa_file="{asym_id}.afa",
    mul_afa_file="{asym_id}.afa",
    training_cutoff="2020-04-30",  # doi.org/10.1038/s41592-023-02086-5 (Methods)
)

RF2NA_LAUNCH_SH = RF2NA.install_dir / "run_RF2NA.sh"

RHOFOLD = Baseline(
    name="rho",
    install_dir=Path("/path/to/RhoFold"),
    out_dir=PREDICTIONS_DIR / "rho",
    job_sh=Path("./scripts/rhofold.sh").resolve(),
    out_pdb="output/unrelaxed_model.pdb",
    afa_file="sequence.a3m",
    mul_afa_file=None,
    training_cutoff="2022-04-13",  # doi.org/10.1038/s41592-024-02487-0 (Results)
)

TRRNA = Baseline(
    name="trRNA",
    install_dir=Path("/path/to/trRosettaRNA_v1.1"),
    out_dir=PREDICTIONS_DIR / "trRNA",
    job_sh=Path("./scripts/trRNA.sh").resolve(),
    out_pdb="model_1.pdb",
    afa_file="sequence.a3m",
    mul_afa_file=None,
    training_cutoff="2022-01-01",  # doi.org/10.1038/s41467-023-42528-4 (Methods)
)

BASELINES = (AF3, NUFOLD, RF2NA, RHOFOLD, TRRNA)
BASELINES = {b.name: b for b in BASELINES}
TRAINING_CUTOFF = max(b.training_cutoff for b in BASELINES.values())


def get_bl_out_pdb(
    bl_name: str, pdb_id: str, chain_id: str, multimer: bool
) -> Tuple[Path, Path]:
    """
    Gets the output PDB file for a given baseline, PDB ID, and asym. chain ID.
    """
    label = "multimers" if multimer else "monomers"
    name = pdb_id.lower() if multimer else f"{pdb_id.lower()}_{chain_id}"
    out_file = BASELINES[bl_name].out_pdb.format(
        name_upper=name.upper(),
        name_lower=name.lower(),
    )

    out_dir = (Path("predictions") / bl_name / label / name).resolve()
    out_file = (out_dir / out_file).resolve()
    return out_dir, out_file


# --- Exporting ---
IDENTIFIER_COLS = [
    "PDB ID",
    "Asym. Chain ID",
    "Auth. Chain ID",
]
MON_EVAL_COLS = [
    f"{bl.upper()} MONOMERS {metric}"
    for bl in BASELINES.keys()
    for metric in ["TM Score", "INF_WC", "INF_NWC"]
]

MUL_EVAL_COLS = [
    f"{bl.upper()} MULTIMERS {metric}"
    for bl, spec in BASELINES.items()
    if spec.mul_afa_file is not None
    for metric in ["TM Score", "INF_WC", "INF_NWC"]
]
MON_EXPORT_COLS = IDENTIFIER_COLS + MON_EVAL_COLS
MUL_EXPORT_COLS = IDENTIFIER_COLS + MUL_EVAL_COLS
