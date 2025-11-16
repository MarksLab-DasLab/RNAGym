#!/usr/bin/env python3

###############################################################################
# `analysis.py`: Code for RNA structural analyses
###############################################################################
from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr
from dataclasses import dataclass
from enum import Enum, auto
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile as NamedTemporaryFile
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import gemmi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RNA_normalizer
from Bio import Align, AlignIO
from Bio.PDB import PDBIO, MMCIFParser
from evcouplings.compare import PDB, Chain, DistanceMap, add_distances
from evcouplings.utils.pipeline import execute

from util import (
    AltID,
    AtomName,
    Baseline,
    Category,
    ChainID,
    Config,
    Fr3dContactID,
    Fr3dResID,
    InsertionCode,
    ModelID,
    PdbID,
    ResID,
    ResName,
    SymmetryOp,
)
from util.structure import ChainType, StructureInfo, get_structure


@dataclass
class Residue:
    """
    Class for identifying and categorizing residues in a PDB structure.
    """

    pdb_id: PdbID
    model_id: ModelID
    chain_id: ChainID
    res_name: ResName
    res_id: ResID  # Possibly non-unique unless paired with the insertion code
    seqres_id: ResID  # Unique, 1-indexed residue index
    atom_name: Optional[AtomName]
    alt_id: Optional[AltID]
    ins_code: Optional[InsertionCode]
    symmetry_op: Optional[SymmetryOp]

    @staticmethod
    def from_fr3d(chain: Chain, residue_id: Fr3dResID) -> Residue:
        """
        Initialize a residue from its Fr3dResID description. See:
          - https://www.bgsu.edu/research/rna/help/rna-3d-hub-help/unit-ids.html
        """
        components = [None if item == "" else item for item in residue_id.split("|")]
        n_components = len(components)

        if n_components not in (5, 8, 9):
            raise ValueError("Fr3d residue ID must have 5, 8, or 9 components")

        components += [None] * (9 - n_components)
        (
            pdb_id,
            model_id,
            chain_id,
            res_name,
            res_id,
            atom_name,
            alt_id,
            ins_code,
            symmetry_op,
        ) = components

        # Identify the unique, sequential residue ID
        resid_query = res_id if ins_code is None else res_id + ins_code
        residues = chain.residues.loc[chain.residues.coord_id == resid_query]
        if len(residues) != 1:
            raise RuntimeError(f"Found 0 or multiple residues matching resid {res_id}")
        seqres_id = residues.iloc[0]["seqres_id"]

        return Residue(
            PdbID(pdb_id),
            ModelID(model_id),
            ChainID(chain_id),
            ResName(res_name),
            ResID(res_id),
            ResID(seqres_id),
            AtomName(atom_name),
            AltID(alt_id),
            InsertionCode(ins_code),
            SymmetryOp(symmetry_op),
        )


@dataclass
class Contact:
    """
    Class for identifying and categorizing contacts in a PDB structure.
    """

    residue_1: Residue
    residue_2: Residue
    label: Category

    @staticmethod
    def from_fr3d(chain: Chain, contact_id: Fr3dContactID, label: Category):
        """
        Initializes a contact from its FR3D representation (a string).

        Parameters:
            chain (Chain):  The chain from which contacts were identified with
              FR3D.
            contact (Fr3dContactID):  The contact identifier assigned by FR3D.
            label (Category):  The contact category assigned by FR3D (e.g., 's35'.
        """
        residue_1_id, residue_2_id, _ = contact_id
        residue_1 = Residue.from_fr3d(chain, residue_1_id)
        residue_2 = Residue.from_fr3d(chain, residue_2_id)

        return Contact(residue_1, residue_2, label)

    def __eq__():
        """
        Check if two Contacts represent the same pair of residues.
        """
        raise NotImplementedError("== not yet implemented for Contact")

    def __hash__():
        """
        Hash for contacts such that any two Contact objects involving the same
        two residues are identical.
        """
        raise NotImplementedError("Hash not yet implemented for Contact")


class State(Enum):
    INIT = auto()
    ANALYZED = auto()


@dataclass
class USAlignResult:
    """
    Class for parsing the results of USAlign.

    Members
    -------
    target_name (str):
        The name of the target PDB.
    reference_name (str):
        The name of the reference PDB.
    tm_score (float):
        The TM score of aligning the target to the reference.
    seq_id (float):
        The sequence ID fraction of the target aligned to the reference.
    """

    target_name: str
    ref_name: str
    tm_score: float
    seq_id: float

    __TARGET_NAME_RE = re.compile(r"^Name of Structure_1:\s+(\S+):", re.MULTILINE)
    __REF_NAME_RE = re.compile(r"^Name of Structure_2:\s+(\S+):", re.MULTILINE)
    __TM_SCORE_RE = re.compile(
        r"^TM-score=\s+(\S+)\s+\(normalized by length of Structure_2:", re.MULTILINE
    )
    __ALIGN_RE = re.compile(
        r"^Aligned length=\s+(\S+),\s+RMSD=\s+(\S+),\s+Seq_ID=n_identical/n_aligned=\s+(\S+)",
        re.MULTILINE,
    )

    def __init__(self, usalign_res: str):
        # Names should be stripped of leading "/" and trailing ".pdb"
        target = self.__TARGET_NAME_RE.search(usalign_res)
        self.target_name = os.path.splitext(os.path.basename(target.group(1).strip()))[
            0
        ]
        ref = self.__REF_NAME_RE.search(usalign_res)
        self.ref_name = os.path.splitext(os.path.basename(ref.group(1).strip()))[0]

        tm = self.__TM_SCORE_RE.search(usalign_res)
        self.tm_score = float(tm.group(1))

        align = self.__ALIGN_RE.search(usalign_res)
        self.seq_id = float(align.group(3))


def get_rf2na_chain_ids(
    rf2na_launch_sh: Union[str, os.PathLike[str]],
) -> Dict[ChainID, ChainID]:
    """
    This function returns a mapping from chain asym. IDs to
    RoseTTAFold2NA output chain IDs.  This is necessary because
    RoseTTAFold2NA does not preserve input chain IDs.  They are
    instead re-labeled as A-Z in the order they are provided.
    """
    with open(rf2na_launch_sh, "r") as f:
        lines = f.read().splitlines()

    cmd_index = next(
        index
        for index, line in enumerate(lines)
        if Path(Config.RF2NA_LAUNCH_SH).name in line
    )

    asym_id_to_rf2na_id = {}
    curr_letter = 0
    labels = []
    for line in lines[cmd_index + 1 :]:
        line = line.strip(" \r\n\\")
        label, fa_file = line.split(":")
        asym_id, _ = fa_file.split(".")
        asym_id_to_rf2na_id[asym_id] = chr(ord("A") + curr_letter)
        labels.append(label)
        curr_letter += 1

    # NOTE(MCA): Unclear why this happens, but this case is flipped from the
    #  usual ordering.
    if labels == ["R", "P"]:
        key1, key2 = list(asym_id_to_rf2na_id.keys())
        asym_id_to_rf2na_id[key1], asym_id_to_rf2na_id[key2] = (
            asym_id_to_rf2na_id[key2],
            asym_id_to_rf2na_id[key1],
        )

    return asym_id_to_rf2na_id


def write_chain_minimal_pdb(
    pdb_file: Path, dst_chain_id: str, src_chain_id: str
) -> Path:
    """
    Takes a PDB file and outputs a minimal PDB containing only the input chain
    ID.
    """
    structure = gemmi.read_structure(str(pdb_file))
    model = structure[0]
    new_structure = gemmi.Structure()
    new_model = gemmi.Model(0)
    new_chain = gemmi.Chain(dst_chain_id)

    old_chain = next(
        (chain for chain in model if chain.name == src_chain_id),
        None,
    )
    new_chain.append_residues(old_chain.whole())
    new_model.add_chain(new_chain)
    new_structure.add_model(new_model)
    pdb_file = pdb_file.with_name(f"{pdb_file.stem}_{dst_chain_id}.pdb")
    new_structure.write_minimal_pdb(str(pdb_file))
    return pdb_file


def InteractionNetworkFidelity(reference_pdb, prediction_pdb):
    """
    Calculates interaction network fidelity, only comparing residues
    that are present and not "N".
    """

    def load_struct(pdb_path, idx=None):
        s = RNA_normalizer.PDBStruct()
        s.load(pdb_path, idx)
        return s, s.raw_sequence()

    # 1) Load both fully (no index yet)
    ref_struct, ref_seq = load_struct(reference_pdb)
    pred_struct, pred_seq = load_struct(prediction_pdb)

    # Create a mapping from resid to resname for the predicted structure
    struct = gemmi.read_pdb(str(prediction_pdb))
    resid_to_resname = {
        r.seqid.num: r.name for chain in struct[0].subchains() for r in chain
    }

    # 2) Load the predicted structure with indices matching the reference
    # NOTE(MCA): The reference resids are guaranteed to align with their
    #   position in the sequence.  For example, if the first resid in the
    #   reference structure is 3 it means the first 2 residues from the
    #   sequence are missing.
    ref_indices = [
        r.pos
        for r in ref_struct.res_list
        if resid_to_resname.get(r.pos, "N") != "N"
        and r.res.resname in {"A", "C", "G", "U"}
    ]

    with NamedTemporaryFile("w") as tmp:
        ref_chain = ref_struct.res_list[0].chain
        pred_chain = pred_struct.res_list[0].chain

        # Reload reference structure
        tmp.write("\n".join(f"{ref_chain}:{i}:1" for i in ref_indices))
        tmp.flush()
        ref_struct, ref_seq = load_struct(reference_pdb, tmp.name)

        # Reload predicted structure
        tmp.seek(0)
        tmp.truncate()
        tmp.write("\n".join(f"{pred_chain}:{i}:1" for i in ref_indices))
        tmp.flush()
        pred_struct, pred_seq = load_struct(prediction_pdb, tmp.name)

    if ref_seq != pred_seq:
        raise ValueError(f"{ref_seq} != {pred_seq}")

    # 3) Compute metrics
    # NOTE(MCA): Stacking interactions are currently separate from INF_NWC
    c = RNA_normalizer.PDBComparer()
    rmsd = c.rmsd(pred_struct, ref_struct)
    INF_ALL = c.INF(pred_struct, ref_struct, "ALL")
    DI_ALL = rmsd / INF_ALL if INF_ALL else None
    INF_WC = c.INF(pred_struct, ref_struct, "PAIR_2D")
    INF_NWC = c.INF(pred_struct, ref_struct, "PAIR_3D")
    INF_STACK = c.INF(pred_struct, ref_struct, "STACK")
    return (rmsd, DI_ALL, INF_ALL, INF_WC, INF_NWC, INF_STACK)


class Analysis:
    """
    Object for initializing and conducting RNA structural analyses.

    Members:
        chain:  The Chain object corresponding to chain_id.
        pdb_id:  The PDB ID under consideration.
        asym_id:  The asym. chain ID of the chain to analyze.
        auth_id:  The auth. chain ID of the chain to analyze.
    """

    chain_id: ChainID
    pdb_id: PdbID
    chain: Chain
    structure: gemmi.Structure
    __contacts: Optional[Dict[Category, Contact]]
    __distance_map: Optional[DistanceMap]  # The DistanceMap retrieved from EVCouplings
    __state: State  # The current state (Analysis is a state machine)
    __seqres_to_index: Dict[
        ResID, int
    ]  # Mapping from seqres ID to its index in __chain.residues

    def __init__(self, pdb_id: str, asym_id: str, auth_id: str):
        self.pdb_id = pdb_id.lower()
        self.asym_id = asym_id
        self.auth_id = auth_id
        self.__contacts = None
        self.__distance_map = None
        self.__state = State.INIT
        _, self.structure = get_structure(self.pdb_id)

        pdb = PDB(
            Config.RCSB_ASSEMBLY_FILE.format(pdb_id=self.pdb_id),
            binary=False,
            keep_full_data=False,
        )
        self.chain = pdb.get_chain(self.asym_id, is_author_id=False)

        ## Since we may select by auth chain, we should ensure that we only
        ## process residues belonging to the polymer entity of interest.
        # target_entity_id = None
        # for _, residue in self.chain.residues.iterrows():
        #    if residue.three_letter_code in Residues.RNA:
        #        target_entity_id = residue.entity_id
        #        break

        ## Drop all atoms and residues belonging to entities other than the
        ## RNA of interest.
        # self.chain.residues = self.chain.residues[
        #    self.chain.residues.entity_id == target_entity_id
        # ].reset_index(drop=True)
        # self.chain.coords = self.chain.coords.loc[
        #    self.chain.coords.residue_index.isin(self.chain.residues.index)
        # ].reset_index(drop=True)

        # Map seqres IDs to their index in the residue table
        self.__seqres_to_index = {}
        for index, residue in self.chain.residues.iterrows():
            self.__seqres_to_index[ResID(residue.seqres_id)] = index

    @property
    def identifier(self) -> Path:
        """
        Returns a unique identifier for this analysis using the asym. chain ID.
        This should match the directory for RNA 3D Bench inputs.
        """
        return Config.SEQUENCE_ID.format(
            pdb_id=self.pdb_id.upper(), chain_id=self.asym_id
        )

    @property
    def prefix(self) -> Path:
        """
        Returns the prefix path for this analysis (i.e., the base directory
        where the analysis inputs are located and where the outputs will be
        stored).
        """
        prefix = Path(Config.get_out_prefix(self.pdb_id, self.asym_id))
        os.makedirs(prefix, exist_ok=True)
        return prefix

    @property
    def fasta_path(self) -> Path:
        """
        Returns the path to the .fasta file for this analysis.
        """
        fasta_path = self.prefix / "rMSA/sequence.fa"
        return fasta_path

    @property
    def sto_path(self) -> Path:
        """
        Returns the path to the .sto file for this analysis.
        """
        sto_path = self.prefix / "rMSA/sequence.sto"

        # Convert to Stockholm from rMSA output `.afa` when needed
        if not os.path.exists(sto_path):
            afa_path = self.prefix / "rMSA/sequence.afa"
            with open(afa_path, "r") as inp, open(sto_path, "w") as out:
                alignment = AlignIO.read(inp, "fasta")

                # Replace the 'query' record ID with the proper identifier
                if not alignment[0].id == "query":
                    raise ValueError(
                        "Expected 'query' to be the first record in {afa_path}"
                    )
                alignment[0].id = self.identifier

                # Convert each record to RNA if it is DNA
                for record in alignment:
                    record.seq = record.seq.transcribe()

                AlignIO.write(alignment, out, "stockholm")

        return sto_path

    def baseline_results(self, baseline: Baseline, multimer: bool) -> Path:
        """
        Check the input baseline's results for this Analysis.

        Parameters
        ----------
        baseline : Baseline
            The baseline to locate.
        multimer : bool
            If True, gets the multimer results for this baseline.  Otherwise,
            gets the monomer results.

        Returns
        -------
        out_dir : Path
            The output directory where the baseline was run.
        out_pdb : Optional[Path]
            The output PDB containing the baseline's prediction.
        """
        out_dir, out_pdb = Config.get_bl_out_pdb(
            baseline.name, self.pdb_id, self.asym_id, multimer
        )

        if not out_pdb.exists():
            return (out_dir, None)

        # Extract single chain from complex if needed
        if multimer:
            target_asym_id = self.asym_id
            # NOTE(MCA): RF2NA does not honor input chain IDs, so it requires
            #  special treatment
            if baseline.name.lower() == "rf2na":
                rf2na_launch_sh = Path(out_dir / "launch.sh")
                rf2na_asym_ids = get_rf2na_chain_ids(rf2na_launch_sh)
                target_asym_id = rf2na_asym_ids[target_asym_id]

            out_pdb = write_chain_minimal_pdb(out_pdb, self.asym_id, target_asym_id)

        return out_dir, out_pdb

    def success(self, baseline: Baseline, multimer: bool) -> bool:
        """
        Returns True if the baseline's results for this Analysis are complete.
        Otherwise, False.
        """
        # NOTE(MCA): A small % of RhoFold runs fail during minimization, but
        #  can still be considered "successful" for the purpose of evaluation.
        #  The minimization process reduces average TM score by about 0.05.
        out_dir, out_pdb = self.baseline_results(baseline, multimer)
        success_file = out_dir / "SUCCESS"
        return (
            (baseline.name == "rho" or success_file.exists())
            and out_pdb is not None
            and out_pdb.exists()
        )

    def usa_result(self, baseline: Baseline, multimer: bool) -> USAlignResult:
        """
        Returns the USAlignResult for the baseline prediction against the known
        structure for this Analysis.
        """
        # Extract the RNA to be evaluated into its own CIF file
        out_dir, out_pdb = self.baseline_results(baseline, multimer)

        # Align the predicted RNA vs. the known target from the PDB
        reference_pdb = Path(
            Config.CHAIN_MINIMAL_PDB_FILE.format(
                pdb_id=self.pdb_id, chain_id=self.asym_id
            )
        )
        usa_cmd = shlex.split(
            f"{Config.TOOLS['usalign']} {str(out_pdb)} {(reference_pdb)}"
        )
        usa_result = subprocess.run(
            usa_cmd, capture_output=True, text=True, check=True
        ).stdout

        return USAlignResult(usa_result)

    def inf(self, baseline: Baseline, multimer: bool) -> Tuple[float, float]:
        """
        Calculates interaction network fidelity (INF).

        Returns
        -------
        (inf_wc, inf_nwc) : Tuple[float, float]
            The Watson-Crick and non-Watson-Crick INF scores for this Analysis.
        """
        # NOTE(MCA): This assumes that usa_results has already been run,
        #  which will extract the appropriate multimer output file.
        out_dir, out_pdb = self.baseline_results(baseline, multimer)
        reference_pdb = Path(
            Config.CHAIN_MINIMAL_PDB_FILE.format(
                pdb_id=self.pdb_id, chain_id=self.asym_id
            )
        )

        # RNA_normalizer requires PDB inputs
        if out_pdb.suffix == ".cif":
            chain_key = f"{self.pdb_id}_{self.asym_id}"
            structure = MMCIFParser(QUIET=True).get_structure(chain_key, out_pdb)

            # Ensure chain IDs are single-character by assigning them to "A"
            # NOTE(MCA): This is acceptable because out_pdb is guaranteed to be
            #   a single chain file.
            for model in structure:
                for chain in model:
                    chain.id = "A"

            io = PDBIO()
            io.set_structure(structure)
            out_pdb = out_pdb.with_suffix(".pdb")
            io.save(str(out_pdb))

        rmsd, di_all, inf_all, inf_wc, inf_nwc, inf_stack = InteractionNetworkFidelity(
            reference_pdb, out_pdb
        )

        return inf_wc, inf_nwc

    def __run_rna_alignment(self):
        """
        Runs the alignment protocol for the input chain.
        """
        # Configure a basic alignment protocol
        # See: https://github.com/debbiemarkslab/EVcouplings/blob/develop/config/sample_config_monomer.txt
        config = {
            "pipeline": "protein_monomer",
            "stages": ["align"],
            "global": {
                "prefix": str(self.prefix),
                "theta": 0.9,  # cluster 90% identity
                "cpu": 4,
                "region": None,  # Entire sequence
                "sequence_id": self.identifier,
                "sequence_file": str(self.fasta_path),
            },
            "align": {
                "protocol": "existing",
                "alphabet": "rna",
                "input_alignment": str(self.sto_path),
                "first_index": 1,
                "compute_num_effective_seqs": False,
                "seqid_filter": None,
                "minimum_sequence_coverage": 50,
                "minimum_column_coverage": 70,
                "extract_annotation": True,
                "sequence_weights": "nogaps",
            },
            "databases": Config.DATABASES,
            "tools": Config.TOOLS,
        }

        print(f"Running alignment protocol for {self.identifier}...")
        execute(**config)

    def __run_rna_couplings(self):
        """
        Runs the couplings protocol for the input chain.
        """
        # Configure a basic alignment protocol
        # See: https://github.com/debbiemarkslab/EVcouplings/blob/develop/config/sample_config_monomer.txt
        config = {
            "pipeline": "protein_monomer",
            "stages": ["couplings"],
            "global": {
                "prefix": str(self.prefix),
                "theta": 0.9,  # cluster 90% identity
                "cpu": 4,  # jackhmmer struggles to exploit more than 4 cores
                "region": None,  # Entire sequence
                "sequence_id": self.identifier,
                "sequence_file": str(self.fasta_path),
                "alignment_file": str(self.sto_path),
            },
            "align": {},  # skipped!
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
                "reuse_ecs": True,
                "min_sequence_distance": 6,
                "scoring_model": "logistic_regression",
            },
            "databases": Config.DATABASES,
            "tools": Config.TOOLS,
        }

        print(f"Running couplings protocol for {self.identifier}...")
        execute(**config)

    def get_index(self, seqres_id: ResID) -> int:
        """
        Gets the integer index of the seqres_id in the residues DataFrame,
        which corresponds to its index in the contact matrix.
        """
        return self.__seqres_to_index[seqres_id]

    def run_ev_couplings(self):
        """
        Runs the stages of the EVCouplings pipeline in order.
        """
        if self.__state != State.INIT:
            raise RuntimeError(
                f"run_ev_couplings called from invalid state ({self.__state.name})"
            )

        self.__run_rna_alignment()
        self.__run_rna_couplings()
        self.__state = State.ANALYZED

    def get_ecs(self) -> pd.DataFrame:
        """
        Returns the evolutionary couplings for the input structure.
        """
        couplings_df = pd.read_csv(
            f"{self.prefix}/couplings/{self.asym_id}_CouplingScores.csv"
        )
        couplings_df = add_distances(couplings_df, self.distance_map)
        return couplings_df

    def get_ali_stats(self) -> pd.DataFrame:
        """
        Returns the alignment statistics for the input structure.
        """
        return pd.read_csv(
            f"{self.prefix}/align/{self.asym_id}_alignment_statistics.csv"
        )

    @property
    def distance_map(self) -> DistanceMap:
        """
        Returns a distance map for the input structure.
        """
        if self.__distance_map is None:
            self.__distance_map = DistanceMap.from_coords(self.chain)
        return self.__distance_map


def prep_fasta(input_fa, output, seq_unmod, I_compatible=True, wrap_col=100) -> str:
    """
    Copies the input MSA into the output directory, replacing nucleotides in
    the query that were identified in the MSA process with Ns when indicated in
    the structure sequence.  Ts in the query are replaced with Us. If the input
    does not exist, the output is a single-sequence MSA containing just the
    query sequence.  Returns the final query sequence.

    Parameters
    ----------
    input_fa : Path
        Path to the input MSA (e.g., .fa, .a3m. .afa)
    output_fa : Path
        Path to the output MSA (extension should match input_fa)
    seq_unmod : str
        The true sequence in the evaluation PDB as one-letter nucleotides.
        This should just be the query sequence, with known nucleotides replaced
        with Ns at respective locations.
    wrap_col : int
        The column to wrap to if the input is a `.a3m` file.  Defaults to 100.
    I_compatible : bool
        If True, allows inosine in the input MSA.  Otherwise, Is are replaced
        with As as they likely originate from A-to-I editing and are
        structurally similar.
    """
    if input_fa.exists():
        with open(input_fa, "r") as f:
            contents = f.read().splitlines()
    else:
        raise ValueError(f"Unable to locate input MSA ({input_fa})")

    seq_unmod = seq_unmod.replace("T", "U")
    if not I_compatible:
        # Replace inosine w/ adenosine (similar structure; likely originates
        # from A-to-I editing)
        seq_unmod = seq_unmod.replace("I", "A")
    with open(output, "w") as f:
        if len(contents) < 2:
            contents = [">query", seq_unmod]

        i = 1
        while i < len(contents) and not contents[i].startswith(">"):
            i += 1
        old_seq = "".join(contents[1:i])
        if len(old_seq) != len(seq_unmod):
            raise ValueError(
                f"MSA query ({old_seq}) and sequence ({seq_unmod}) length do not match"
            )

        # Retain the MSA's best guess sequence if the output MSA contains Ns
        seq_unmod = old_seq.replace("T", "U").replace("N", "-")

        # Only wrap lines if output extension is .a3m
        rest_of_msa = contents[i:]
        if output.suffix == ".a3m" and wrap_col is not None:
            wrapped_seq = [
                seq_unmod[x : x + wrap_col] for x in range(0, len(seq_unmod), wrap_col)
            ]
            contents = [contents[0]] + wrapped_seq
        else:
            contents = [contents[0], seq_unmod]

        # Unwrap ".a3m" if needed (required by trRNA)
        if output.suffix == ".a3m" and wrap_col is None:
            j = 0
            while j < len(rest_of_msa):
                line = rest_of_msa[j].rstrip()
                # If it's a header line, just move on
                if line.startswith(">"):
                    j += 1
                    continue
                # Otherwise, accumulate all subsequent lines until next header
                k = j + 1
                while k < len(rest_of_msa) and not rest_of_msa[k].startswith(">"):
                    rest_of_msa[j] = rest_of_msa[j].rstrip() + rest_of_msa[k].strip()
                    del rest_of_msa[k]
                j += 1

        contents += rest_of_msa

        # Replace Ts with Us
        for idx, line in enumerate(contents):
            if not line.startswith(">"):
                contents[idx] = line.replace("T", "U")

        f.seek(0)
        f.write("\n".join(contents))
        f.truncate()

        return seq_unmod


def prep_af3(out_dir=Config.AF3.out_dir):
    """
    Writes run configurations for AlphaFold3.

    Parameters
    ----------
    out_dir : str
        The output directory to place the run configuration.
    """
    out_path = Path(out_dir)
    mon_path = (out_path / "monomers").resolve()
    mon_path.mkdir(parents=True, exist_ok=True)
    mon_df = pd.read_csv(Config.MONOMER_CSV)
    mul_df = pd.read_csv(Config.MULTIMER_CSV)

    # --- Process monomer runs ---
    # For each monomer (i.e. each row in mon_df) we create a configuration that
    # provides both the sequence file and the pre-computed MSA file.
    for _, row in mon_df.iterrows():
        # Extract minimal fields from CSV
        pdb_id = row["PDB ID"].lower()
        asym_chain_id = row["Asym. Chain ID"]
        seq_unmodified = row["Sequence (unmod.)"]
        out_dir = mon_path / f"{pdb_id}_{asym_chain_id}"

        # Don't overwrite successful run configs
        if (out_dir / "SUCCESS").exists():
            continue
        print(f"Preparing {pdb_id}_{asym_chain_id}")

        # Create top-level AF3 input e.g., name = "AF3_1ABC_A"
        sequence_a3m = Path(f"./out/{pdb_id}/{asym_chain_id}/rMSA/sequence.a3m")
        out_a3m = out_dir / "sequence.a3m"
        out_fp = out_dir / "config.json"
        seq_unmod = prep_fasta(sequence_a3m, out_a3m, seq_unmodified)
        config = {
            "name": f"{pdb_id}_{asym_chain_id}",
            "modelSeeds": [0],  # or any seeds you like, e.g. [42, 123].
            "sequences": [
                {
                    "rna": {
                        "id": str(asym_chain_id),
                        "sequence": str(seq_unmod),
                        "unpairedMsaPath": str(out_a3m),
                    }
                }
            ],
            "dialect": "alphafold3",
            "version": 2,
        }

        # Write the JSON file named "<pdb_id>_<asym_chain_id>.json"
        # under the out_dir directory.
        out_fp.parent.mkdir(exist_ok=True)
        with open(out_fp, "w") as f:
            json.dump(config, f, indent=2)

    # -----------------------
    # Process multimer runs
    # -----------------------
    #
    # For multimer runs we group the multimer CSV by PDB ID (so one run per PDB).
    # Then for each chain in that PDB we supply the sequence file; however, if the
    # chain is RNA we also provide the MSA file (using the same sub-directory pattern).
    mul_path = (out_path / "multimers").resolve()
    mul_path.mkdir(exist_ok=True)
    grouped = mul_df.groupby("PDB ID")

    # A small helper map from ChainType -> AF3 JSON key.
    chain_type_to_af3 = {
        ChainType.RNA: "rna",
        ChainType.PROTEIN: "protein",
        ChainType.DNA: "dna",
    }

    # We iterate over each group. If a PDB is present in the CSV but not in the structure
    # data, or vice versa, you may want to handle that carefully (try/except).
    Config.HYBRID_CUTOFF = 0.5  # Assign hybrid chains to the dominant type
    for pdb_id, group_df in grouped:
        pdb_id_lower = pdb_id.lower()

        # Don't overwrite successful run configs
        out_dir = mul_path / f"{pdb_id_lower}"
        if (out_dir / "SUCCESS").exists():
            continue
        print(f"Preparing {pdb_id} multimer...")

        # Collect a list of chain specification dicts for the AF3 JSON
        info = StructureInfo.from_pdb_id(pdb_id, {"multimer"})
        sequences_list = []

        # Turn the group_df into a lookup dict for MSA path usage
        # Key = Asym. Chain ID, Value = the row from mul_df
        csv_rows = {}
        for _, row in group_df.iterrows():
            # e.g. row["Asym. Chain ID"] => "A"
            asym_id = row["Asym. Chain ID"]
            pdb_id = row["PDB ID"].lower()
            csv_rows[f"{pdb_id}_{asym_id}"] = row

        # info.assembly[0] is the first model
        model = info.assembly[0]

        # We iterate over each chain in that model that StructureInfo recognized.
        # Alternatively, we could iterate over info.chain_types.keys().
        for chain in model.subchains():
            # chain_id might be "A-1", "B-2", "B", etc
            raw_chain_id = chain.subchain_id()
            chain_id, sym_exp_n = (
                raw_chain_id.split("-") if "-" in raw_chain_id else (raw_chain_id, None)
            )

            # Repeat chain ID sym_exp_n - 1 times ("D" stands for dash, for uniqueness)
            # NOTE(MCA): In rare cases not seen in the RNAFold evaluation sets,
            #   this could conflict with other chains the PDB.
            sym_exp = "" if sym_exp_n is None else f"D{chain_id * (int(sym_exp_n) - 1)}"

            # If structure info says this chain ID is not recognized or was skipped,
            # skip or handle gracefully:
            if chain_id not in info.chain_types:
                continue

            # Determine chain type from structure info
            ctype = info.chain_types[chain_id]
            if ctype not in chain_type_to_af3:
                continue
            af3_chain_type = chain_type_to_af3[ctype]  # "rna", "protein", ...

            # Get the unmodified sequence from structure info
            seq_unmod = info.sequences_unmod[chain_id]

            # Build the chain sub-dict for AF3
            chain_dict = {
                af3_chain_type: {
                    "id": f"{chain_id}{sym_exp}",
                    "sequence": str(seq_unmod),
                }
            }

            sequence_a3m = Path(f"./out/{pdb_id_lower}/{chain_id}/rMSA/sequence.a3m")
            out_a3m = out_dir / f"sequence_{chain_id}.a3m"
            out_fp = out_dir / "config.json"
            out_fp.parent.mkdir(exist_ok=True)

            # If this chain is RNA *and* is in complex.csv then we have a custom MSA
            csv_key = f"{pdb_id_lower}_{chain_id}"
            if ctype == ChainType.RNA and csv_key in csv_rows:
                seq_unmod = prep_fasta(sequence_a3m, out_a3m, str(seq_unmod))
                chain_dict[af3_chain_type]["sequence"] = seq_unmod
                chain_dict[af3_chain_type]["unpairedMsaPath"] = str(out_a3m)

            # Append to the sequences list
            sequences_list.append(chain_dict)

        # Now build the final JSON dict for this PDB
        config = {
            "name": f"{pdb_id_lower}",
            "modelSeeds": [0],
            "sequences": sequences_list,
            "dialect": "alphafold3",
            "version": 2,
        }

        # Write the JSON file
        with open(out_fp, "w") as f:
            json.dump(config, f, indent=2)

    print(f"Done! Wrote multimer/monomer AF3 JSONs to {out_path}")


def prep_rf2na(out_dir=Config.RF2NA.out_dir):
    """
    Writes run configurations for RoseTTAFold2NA.

    Parameters
    ----------
    out_dir : str
        The output directory to place the run configuration.
    """
    out_path = Path(out_dir)
    mon_path = (out_path / "monomers").resolve()
    mon_path.mkdir(parents=True, exist_ok=True)
    mon_df = pd.read_csv(Config.MONOMER_CSV)
    mul_df = pd.read_csv(Config.MULTIMER_CSV)

    # --- Process monomer runs ---
    # For each monomer (i.e. each row in mon_df) we create a configuration that
    # provides both the sequence file and the pre-computed MSA file.
    for _, row in mon_df.iterrows():
        # Extract minimal fields from CSV
        pdb_id = row["PDB ID"].lower()
        asym_chain_id = row["Asym. Chain ID"]
        seq_unmodified = row["Sequence (unmod.)"]
        out_dir = mon_path / f"{pdb_id}_{asym_chain_id}"
        out_dir.mkdir(exist_ok=True)

        # Don't overwrite successful run configs
        if (out_dir / "SUCCESS").exists():
            continue
        print(f"Preparing {pdb_id}_{asym_chain_id}")

        # Create top-level RF2NA input
        sequence_afa = Path(f"./out/{pdb_id}/{asym_chain_id}/rMSA/sequence.afa")
        sequence_fa = out_dir / f"{asym_chain_id}.fa"
        out_afa = out_dir / f"{asym_chain_id}.afa"
        out_fp = out_dir / "launch.sh"

        # Write sequence fasta and afa
        seq_unmodified = prep_fasta(
            sequence_afa, out_afa, seq_unmodified, I_compatible=False
        )
        with open(sequence_fa, "w") as f:
            f.write(f">query\n{seq_unmodified}\n")

        cmd = [
            "#!/usr/bin/env bash",
            "",
            f"{Config.RF2NA_LAUNCH_SH} . \\",
            f"    R:{asym_chain_id}.fa",
        ]

        # Write the JSON file named "<pdb_id>_<asym_chain_id>.json"
        # under the out_dir directory.
        out_fp.parent.mkdir(exist_ok=True)
        with open(out_fp, "w") as f:
            f.write("\n".join(cmd) + "\n")
        os.chmod(out_fp, os.stat(out_fp).st_mode | stat.S_IEXEC)

    # -----------------------
    # Process multimer runs
    # -----------------------
    # For multimer runs we group the multimer CSV by PDB ID (so one run per PDB).
    # Then for each chain in that PDB we supply the sequence file; however, if the
    # chain is RNA we also provide the MSA file (using the same sub-directory pattern).
    mul_path = (out_path / "multimers").resolve()
    mul_path.mkdir(exist_ok=True)
    grouped = mul_df.groupby("PDB ID")

    # A small helper map from ChainType -> RF2NA type
    chain_type_to_rf2na = {
        ChainType.RNA: "R",
        ChainType.PROTEIN: "P",
        ChainType.DNA: "S",  # S := single-stranded DNA
    }

    # We iterate over each group. If a PDB is present in the CSV but not in the structure
    # data, or vice versa, you may want to handle that carefully (try/except).
    Config.HYBRID_CUTOFF = 0.5  # Assign hybrid chains to the dominant type
    for pdb_id, group_df in grouped:
        pdb_id_lower = pdb_id.lower()

        # Don't overwrite successful run configs
        out_dir = mul_path / f"{pdb_id_lower}"
        if (out_dir / "SUCCESS").exists():
            continue
        print(f"Preparing {pdb_id} multimer...")

        info = StructureInfo.from_pdb_id(pdb_id, {"multimer"})
        chain_args = []

        # Turn the group_df into a lookup dict for MSA path usage
        # Key = Asym. Chain ID, Value = the row from mul_df
        csv_rows = {}
        for _, row in group_df.iterrows():
            # e.g. row["Asym. Chain ID"] => "A"
            asym_id = row["Asym. Chain ID"]
            pdb_id = row["PDB ID"].lower()
            csv_rows[f"{pdb_id}_{asym_id}"] = row

        # info.assembly[0] is the first model
        model = info.assembly[0]

        # We iterate over each chain in that model that StructureInfo recognized.
        # Alternatively, we could iterate over info.chain_types.keys().
        for chain in model.subchains():
            # chain_id might be "A-1", "B-2", "B", etc
            raw_chain_id = chain.subchain_id()
            chain_id, sym_exp_n = (
                raw_chain_id.split("-") if "-" in raw_chain_id else (raw_chain_id, None)
            )
            # Repeat chain ID sym_exp_n - 1 times
            sym_exp = "" if sym_exp_n is None else chain_id * (int(sym_exp_n) - 1)

            # If structure info says this chain ID is not recognized or was skipped,
            # skip or handle gracefully:
            if chain_id not in info.chain_types:
                continue

            # Determine chain type from structure info
            ctype = info.chain_types[chain_id]
            if ctype not in chain_type_to_rf2na:
                continue

            prefix = chain_type_to_rf2na[ctype]

            # Get the unmodified sequence from structure info
            seq_unmod = info.sequences_unmod[chain_id]

            sequence_fa = out_dir / f"{chain_id}{sym_exp}.fa"
            sequence_afa = Path(f"./out/{pdb_id_lower}/{chain_id}/rMSA/sequence.afa")
            out_afa = out_dir / f"{chain_id}{sym_exp}.afa"
            out_fp = out_dir / "launch.sh"
            out_fp.parent.mkdir(exist_ok=True)

            # If this chain is RNA *and* is in complex.csv then we have a custom MSA
            csv_key = f"{pdb_id_lower}_{chain_id}"
            if ctype == ChainType.RNA and csv_key in csv_rows:
                seq_unmodified = prep_fasta(
                    sequence_afa, out_afa, str(seq_unmod), I_compatible=False
                )

            # Write sequence fasta and afa
            with open(sequence_fa, "w") as f:
                f.write(f">query\n{seq_unmod}\n")

            # Append to the sequences list
            chain_args.append(f"{prefix}:{sequence_fa.name}")

        cmd = [
            "#!/usr/bin/env bash",
            "",
            f"{Config.RF2NA_LAUNCH_SH} . \\",
        ]

        for i, arg in enumerate(chain_args):
            slash = "" if (i + 1 == len(chain_args)) else "\\"
            cmd.append(f"    {arg} {slash}")

        # Write the launch script
        with open(out_fp, "w") as f:
            f.write("\n".join(cmd) + "\n")
        os.chmod(out_fp, os.stat(out_fp).st_mode | stat.S_IEXEC)

    print(f"Done! Wrote all multimer/monomer RF2NA launch.sh to {out_path}")


def prep_rhofold(out_dir=Config.RHOFOLD.out_dir):
    """
    Writes run configurations for RhoFold+.

    Parameters
    ----------
    out_dir : str
        The output directory to place the run configuration.
    """
    out_path = Path(out_dir)
    mon_path = (out_path / "monomers").resolve()
    mon_path.mkdir(parents=True, exist_ok=True)
    mon_df = pd.read_csv(Config.MONOMER_CSV)

    # --- Process monomer runs ---
    for _, row in mon_df.iterrows():
        # Extract minimal fields from CSV
        pdb_id = row["PDB ID"].lower()
        asym_chain_id = row["Asym. Chain ID"]
        seq_unmodified = row["Sequence (unmod.)"]
        out_dir = mon_path / f"{pdb_id}_{asym_chain_id}"

        # Don't overwrite successful run configs
        if (out_dir / "SUCCESS").exists():
            continue
        print(f"Preparing {pdb_id}_{asym_chain_id}")

        sequence_a3m = Path(f"./out/{pdb_id}/{asym_chain_id}/rMSA/sequence.a3m")
        out_a3m = out_dir / "sequence.a3m"
        out_fp = out_dir / "sequence.fa"

        out_fp.parent.mkdir(exist_ok=True)
        seq_unmodified = prep_fasta(sequence_a3m, out_a3m, seq_unmodified)
        with open(out_fp, "w") as f:
            f.write(">query\n")
            f.write(f"{seq_unmodified}\n")


def prep_nufold(out_dir=Config.NUFOLD.out_dir):
    """
    Writes run configurations for nufold.

    Parameters
    ----------
    out_dir : str
        The output directory to place the run configuration.
    """
    out_path = Path(out_dir)
    mon_path = (out_path / "monomers").resolve()
    mon_path.mkdir(parents=True, exist_ok=True)
    mon_df = pd.read_csv(Config.MONOMER_CSV)

    # --- Process monomer runs ---
    for _, row in mon_df.iterrows():
        # Extract minimal fields from CSV
        pdb_id = row["PDB ID"].lower()
        asym_chain_id = row["Asym. Chain ID"]
        seq_unmodified = row["Sequence (unmod.)"]
        out_dir = mon_path / f"{pdb_id}_{asym_chain_id}"

        # Don't overwrite successful run configs
        if (out_dir / "SUCCESS").exists():
            continue

        print(f"Preparing {pdb_id}_{asym_chain_id}")

        chain_key = f"{pdb_id.upper()}_{asym_chain_id}"
        input_dir = out_dir / "input" / chain_key
        sequence_a3m = Path(f"./out/{pdb_id}/{asym_chain_id}/rMSA/sequence.a3m")
        in_a3m = input_dir / f"{chain_key}.a3m"
        in_fa = input_dir / f"{chain_key}.fasta"
        in_ss = input_dir / f"{chain_key}.ipknot.ss"

        in_fa.parent.mkdir(exist_ok=True, parents=True)
        seq_unmodified = prep_fasta(sequence_a3m, in_a3m, seq_unmodified)
        with open(in_fa, "w") as f:
            f.write(f">{chain_key}\n")
            f.write(f"{seq_unmodified}\n")

        # Generate IPKnot SS structure
        ipknot_sh = Path(
            f"{Config.BASELINES['nu'].install_dir}/ipknot-1.1.0-x86_64-linux/ipknot"
        ).resolve()
        ipknot_cmd = shlex.split(f"{str(ipknot_sh)} {str(in_fa)}")
        with open(in_ss, "w") as f:
            subprocess.run(ipknot_cmd, stdout=f, check=True)


def prep_trRNA(out_dir=Config.TRRNA.out_dir):
    """
    Writes run configurations for trRosettaRNA.

    Parameters
    ----------
    out_dir : str
        The output directory to place the run configuration.
    """
    out_path = Path(out_dir)
    mon_path = (out_path / "monomers").resolve()
    mon_path.mkdir(parents=True, exist_ok=True)
    mon_df = pd.read_csv(Config.MONOMER_CSV)

    # --- Process monomer runs ---
    for _, row in mon_df.iterrows():
        # Extract minimal fields from CSV
        pdb_id = row["PDB ID"].lower()
        asym_chain_id = row["Asym. Chain ID"]
        seq_unmodified = row["Sequence (unmod.)"]
        out_dir = mon_path / f"{pdb_id}_{asym_chain_id}"

        # Don't overwrite successful run configs
        if (out_dir / "SUCCESS").exists():
            continue

        print(f"Preparing {pdb_id}_{asym_chain_id}")

        chain_key = f"{pdb_id.lower()}_{asym_chain_id}"
        sequence_a3m = Path(f"./out/{pdb_id}/{asym_chain_id}/rMSA/sequence.a3m")
        out_fa = out_dir / "sequence.fa"
        out_a3m = out_dir / "sequence.a3m"
        out_a3m.parent.mkdir(exist_ok=True)

        seq_unmodified = prep_fasta(
            sequence_a3m, out_a3m, seq_unmodified, wrap_col=None
        )
        with open(out_fa, "w") as f:
            f.write(f">{chain_key}\n")
            f.write(f"{seq_unmodified}\n")


def __get_max_seq_id(args: Iterable[Any]):
    """
    Helper function for identifying maximum sequence identities.
    """
    row, target_seqs, pdb_id_to_date = args
    index, pdb_id, auth_id, published, query_seq = row
    print(f"Processing \"5'-{query_seq[0:15]}...-3'\"...")

    # See: https://biopython.org/docs/dev/Tutorial/chapter_pairwise.html
    aligner = Align.PairwiseAligner(
        match_score=1.0, mismatch_score=-2.0, gap_score=-2.5
    )

    # Identify the hit with the best sequence %ID
    max_seq_results = {
        b: (0.0, None, "0000-00-00")  # pct_id, target_name, published
        for b in Config.BASELINES.values()
    }

    for (
        target_pdb_id,
        target_auth_id,
        target_published,
        target_seq,
    ) in target_seqs:
        alignment = aligner.align(query_seq, target_seq)[0]
        pct_id = alignment.counts().identities / alignment.length
        coverage = alignment.length / max(len(query_seq), len(target_seq))
        pct_id *= coverage  # penalize %IDs without 100% coverage

        for bl, (max_pct_id, _, _) in max_seq_results.items():
            if target_published <= bl.training_cutoff:
                if pct_id > max_pct_id:
                    max_id_target = f"{target_pdb_id}_{target_auth_id}"
                    max_seq_results[bl] = (pct_id, max_id_target, target_published)

    return index, max_seq_results


def __get_max_tm_results(args: Iterable[Any]):
    """
    Helper function for identifying maximum sequence identities.
    """
    index, pdb_id, asym_id, pdb_id_to_date = args
    chain_id = f"{pdb_id.lower()}_{asym_id}"
    query_file = Path(f"{Config.get_out_prefix(pdb_id.lower(), asym_id)}/{asym_id}.pdb")
    cached_out_file = Path(f"./{Config.CHAINS_DIR}/{chain_id}.out")
    chains_list_file = Path(Config.CHAINS_LIST_FILE)
    chains_dir = Path(Config.CHAINS_DIR)
    cached_out_file.touch(exist_ok=True)

    # Run USAlign if needed, or use cached output
    print(f"Processing {chain_id}...")
    with open(cached_out_file, "r+") as f:
        usa_results = f.read()
        _, _, last_line = usa_results.rstrip().rpartition("\n")
        if (
            usa_results == ""
            or last_line.startswith("#Total CPU time is  0.00 seconds")
            or not last_line.startswith("#Total CPU time is")
        ):
            # Trailing slash after chains_dir is required
            usa_cmd = (
                f"{Config.TOOLS['usalign']} -dir1 '{chains_dir}/' '{chains_list_file}' "
                f"'{query_file}'"
            )
            print(f"Running {usa_cmd}")

            # MUST run as a shell instead of the usual shlex.split(usa_cmd),
            # shell=False
            usa_result = subprocess.run(
                usa_cmd, capture_output=True, text=True, check=True, shell=True
            )
            usa_results = usa_result.stdout
            usa_errors = usa_result.stderr

            if usa_errors.strip() != "":
                print("---USA Errors---", file=sys.stderr)
                print(usa_errors, file=sys.stderr)

            f.seek(0)
            f.write(usa_results)
            f.truncate()

        usa_results = usa_results.split("\n\n\n")
        try:
            usa_results = [USAlignResult(res) for res in usa_results]
        except Exception as e:
            print(f"`{usa_cmd}` failed with results: {usa_results}")
            raise e

    # Identify the hit with the best TM score
    max_tm_results = {
        b: (0.0, None)  # max_tm, max_tm_result
        for b in Config.BASELINES.values()
    }

    for usa_result in usa_results:
        for bl, (max_tm, max_tm_result) in max_tm_results.items():
            pdb_id, asym_id = usa_result.target_name.split("_")
            published = pdb_id_to_date[pdb_id.upper()]
            if published <= bl.training_cutoff:
                if usa_result.tm_score > max_tm:
                    max_tm_results[bl] = (usa_result.tm_score, usa_result)

    return index, max_tm_results


def prep_usalign(
    cutoff_col="Published",
    cutoff=Config.TRAINING_CUTOFF,
):
    """
    Prepares inputs for USAlign.

    Parameters
    ----------
    cutoff_col (str):
        The column that will be filtered for values >= cutoff.
    cutoff (str):
        The date cutoff to use, formatted as a string (YYYY-MM-DD).  Only rows
        where cutoff_col is less than or equal to this cutoff will be
        considered as possible homology targets.
    """
    rcsb_df = pd.read_csv(
        "./annotated_chain_ids.csv",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    rcsb_df = rcsb_df[rcsb_df[cutoff_col] <= cutoff]

    # Symlink all RNA chains prior to the cutoff to CHAINS_DIR
    rna_chains = rcsb_df[["PDB ID", "Asym. Chain ID", "Published"]]
    rna_chains = [
        (pdb_id.lower(), asym_id)
        for pdb_id, asym_id, _ in rna_chains.itertuples(index=False)
    ]
    os.makedirs(Config.CHAINS_DIR, exist_ok=True)
    with open(Config.CHAINS_LIST_FILE, "w") as f:
        for pdb_id, asym_id in rna_chains:
            chain_id = f"{pdb_id.lower()}_{asym_id}"
            prefix = Config.get_out_prefix(pdb_id, asym_id)
            source_pdb = Path(f"{prefix}/{asym_id}.pdb").resolve()
            target_pdb = Path(f"{Config.CHAINS_DIR}/{chain_id}.pdb").resolve()
            if not target_pdb.exists():
                target_pdb.symlink_to(source_pdb)
            f.write(f"{chain_id}.pdb\n")


def add_tm_id(
    chains: pd.DataFrame,
    cutoff_col="Published",
    cutoff=Config.TRAINING_CUTOFF,
):
    """
    Adds a column to the input DataFrame representing the maximum TM identity
    of each row/chain to any RNA chain in RCSB published prior to the input
    cutoff.  The input DataFrame is modified in place.  The target homologs and
    their TM score are added to the columns "TM Homolog" and "TM Homolog
    Score", respectively.

    Parameters
    ----------
    chains (pd.DataFrame):
        The chains that will be checked for homology to RCSB.
    cutoff_col (str):
        The column that will be filtered for values >= cutoff.
    cutoff (str):
        The date cutoff to use, formatted as a string (YYYY-MM-DD).  Only rows
        where cutoff_col is less than or equal to this cutoff will be
        considered as possible homology targets.
    """
    rcsb_df = pd.read_csv(
        "./annotated_chain_ids.csv",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

    rcsb_df = rcsb_df[rcsb_df[cutoff_col] <= cutoff]
    chains_list_file = Path(Config.CHAINS_LIST_FILE)
    if not chains_list_file.exists():
        raise RuntimeError(
            "Attempted to call add_tm_id without first calling prep_usalign"
        )

    pdb_id_to_date = rcsb_df.set_index("PDB ID")[cutoff_col].to_dict()
    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                __get_max_tm_results,
                (
                    (index, pdb_id, asym_id, pdb_id_to_date)
                    for index, pdb_id, asym_id in chains[
                        ["PDB ID", "Asym. Chain ID"]
                    ].itertuples(index=True, name=None)
                ),
            )
        )

    # Add max TM homologs for each baseline
    for index, baseline_results in results:
        for bl, (_, max_tm_result) in baseline_results.items():
            new_col = f"{bl.name.upper()} TM Homolog"
            pdb_id, asym_id = max_tm_result.target_name.split("_")
            max_tm_chain = rcsb_df.query(
                f'`PDB ID` == "{pdb_id.upper()}" and `Asym. Chain ID` == "{asym_id}"'
            )
            chains.loc[index, new_col] = (
                f"{pdb_id.lower()}_{max_tm_chain['Auth. Chain ID'].iloc[0]}"
            )
            chains.loc[index, f"{new_col} Date"] = max_tm_chain["Published"].iloc[0]
            chains.loc[index, f"{new_col} Rfam"] = max_tm_chain["Rfam"].iloc[0]
            chains.loc[index, f"{new_col} Score"] = f"{max_tm_result.tm_score:.5f}"


def add_seq_id(
    chains: pd.DataFrame,
    seq_col="Sequence (unmod.)",
    cutoff_col="Published",
    cutoff=Config.TRAINING_CUTOFF,
):
    """
    Adds a column to the input DataFrame representing the maximum sequence
    identity of each row/chain to any RNA chain in RCSB published prior to the
    input cutoff.  The input DataFrame is modified in place.  The target
    homologs and their % identity are added to the columns "Sequence Homolog"
    and "Sequence Homolog %id", respectively.

    Parameters
    ----------
    chains (pd.DataFrame):
        The chains that will be checked for homology to RCSB.
    seq_col (str):
        The column containing the input chains' sequences.
    cutoff_col (str):
        The column that will be filtered for values >= cutoff.
    cutoff (str):
        The date cutoff to use, formatted as a string (YYYY-MM-DD).  Only rows
        where cutoff_col is less than or equal to this cutoff will be
        considered as possible homology targets.
    """
    rcsb_df = pd.read_csv(
        "./annotated_chain_ids.csv",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )
    rcsb_df = rcsb_df[rcsb_df[cutoff_col] <= cutoff]
    cols = ["PDB ID", "Auth. Chain ID", "Published", seq_col]
    target_seqs = rcsb_df[cols]

    pdb_id_to_date = rcsb_df.set_index("PDB ID")[cutoff_col].to_dict()
    with ProcessPoolExecutor() as executor:
        results = list(
            executor.map(
                __get_max_seq_id,
                (
                    (
                        row,
                        target_seqs.itertuples(index=False, name=None),
                        pdb_id_to_date,
                    )
                    for row in chains[cols].itertuples(index=True, name=None)
                ),
            )
        )

    # Add max sequence %ID homologs for each baseline
    for index, baseline_results in results:
        for bl, (pct_id, target_name, published) in baseline_results.items():
            new_col = f"{bl.name.upper()} Sequence Homolog"
            chains.loc[index, f"{new_col}"] = target_name
            chains.loc[index, f"{new_col} Date"] = published
            chains.loc[index, f"{new_col} %id"] = f"{pct_id:.5f}"


def assign_cluster_0(chains: pd.DataFrame):
    """
    Assigns component 0 chains (those with no Rfam E-value <= 1.0) to their
    most likely cluster (the Rfam with minimum E-value > 1.0, if it exists).
    """
    # NOTE(MCA): It is easier to do this now than at the clustering stage
    #   because allowing Rfam hits with E-values >= 1.0 to form edges causes
    #   the Rfam graph to be fully connected.
    mask = (
        pd.to_numeric(chains["Rfam E-value"], errors="coerce").fillna(float("inf"))
        >= Config.RFAM_BAD_CUTOFF
    )
    chains.loc[mask, "Rfam Cluster"] = chains.loc[mask, "Rfam"].map(
        Config.RFAM_COMPONENTS
    )

    # Assign missing Rfams to new components
    max_comp_id = max(Config.COMPONENT_RFAMS.keys())
    mask = pd.isna(chains["Rfam Cluster"]) & pd.notna(chains["Rfam"])
    missing_rfams = chains.loc[mask, "Rfam"].unique()
    new_comp_ids = range(max_comp_id + 1, max_comp_id + 1 + len(missing_rfams))
    new_rfam_components = dict(zip(missing_rfams, new_comp_ids))
    new_component_rfams = {
        comp_id: [rfam_name] for comp_id, rfam_name in zip(new_comp_ids, missing_rfams)
    }
    chains.loc[mask, "Rfam Cluster"] = (
        chains.loc[mask, "Rfam"]
        .map(Config.RFAM_COMPONENTS)
        .fillna(chains.loc[mask, "Rfam"].map(new_rfam_components))
        .astype(int)
    )
    Config.COMPONENT_RFAMS.update(new_component_rfams)


# --- matplotlib styles ---
dracula_blue = "#6272a4"
dracula_orange = "#ffb86c"
plt.rcParams["font.family"] = "sans"
plt.rcParams["font.size"] = 14
plt.rcParams["mathtext.fontset"] = "stixsans"
axis_label_font_size = 22


def get_baseline_scores(print_status=False, ignore_cache=False) -> pd.DataFrame:
    """
    Checks the progress and reports statistics for baseline evaluations.
    """
    mon_analyzed = (
        Config.MONOMER_ANALYZED_CSV.exists()
        and Config.MONOMER_ANALYZED_CSV.stat().st_size > 0
    )
    mul_analyzed = (
        Config.MULTIMER_ANALYZED_CSV.exists()
        and Config.MULTIMER_ANALYZED_CSV.stat().st_size > 0
    )

    # Use cached output if possible
    if mon_analyzed and mul_analyzed and not ignore_cache:
        mon_df = pd.read_csv(Config.MONOMER_ANALYZED_CSV)
        mul_df = pd.read_csv(Config.MULTIMER_ANALYZED_CSV)
    # Otherwise, calculate TM scores for each baseline
    else:
        # Delete any cached MCAnnotate output files
        for root, dirs, files in itertools.chain(
            os.walk("out/"), os.walk("predictions/")
        ):
            for file in files:
                if file.endswith(".mcout"):
                    os.remove(os.path.join(root, file))

        # Process monomer and multimer CSVs
        mon_df, mul_df = (
            pd.read_csv(Config.MONOMER_CSV),
            pd.read_csv(Config.MULTIMER_CSV),
        )
        datasets = [("monomers", mon_df), ("multimers", mul_df)]
        for label, dataset in datasets:
            for bl_name, bl in Config.BASELINES.items():
                print(f"Processing {bl_name} {label.lower()}...")
                is_multimer = label == "multimers"
                if is_multimer and bl.mul_afa_file is None:
                    continue

                for index, row in dataset.iterrows():
                    pdb_id = row["PDB ID"].lower()
                    asym_id = row["Asym. Chain ID"]
                    auth_id = row["Auth. Chain ID"]
                    a = Analysis(pdb_id=pdb_id, asym_id=asym_id, auth_id=auth_id)

                    # Check if run completed successfully
                    if a.success(bl, is_multimer):
                        bl_key = f"{bl_name.upper()} {label.upper()}"
                        # TM Score
                        dataset.loc[index, f"{bl_key} TM Score"] = a.usa_result(
                            bl, is_multimer
                        ).tm_score

                        # INF
                        with redirect_stderr(StringIO()):
                            dataset.loc[
                                index, [f"{bl_key} INF_WC", f"{bl_key} INF_NWC"]
                            ] = a.inf(bl, is_multimer)

                        # Add N_eff and N_ecs column
                        align_df = a.get_ali_stats()
                        dataset.loc[index, "N_eff"] = align_df.loc[0, "N_eff"]

                        # Add ECs
                        couplings_df = a.get_ecs()
                        dataset.loc[index, "N_ecs"] = len(couplings_df)
                        dataset.loc[index, "N_correct_ecs"] = (
                            couplings_df.loc[:, "dist"] <= 5.0
                        ).sum()
                        couplings_df = couplings_df[couplings_df["probability"] >= 0.50]
                        dataset.loc[index, "N_correct_high_prob_ecs"] = (
                            couplings_df.loc[:, "dist"] <= 5.0
                        ).sum()
                        couplings_df = couplings_df[couplings_df["probability"] >= 0.05]
                        dataset.loc[index, "N_candidate_ecs"] = (
                            couplings_df.loc[:, "dist"] <= 5.0
                        ).sum()

        # Add additional metrics
        eps = 1e-10
        for df in [mon_df, mul_df]:
            df = df[df["N_eff"].fillna(0) > 0]
            df["log_N_eff"] = np.log(mon_df["N_eff"] + eps)
            df["log_N_correct_ecs"] = np.log(mon_df["N_correct_ecs"] + eps)
            df["log_N_correct_high_prob_ecs"] = np.log(
                mon_df["N_correct_high_prob_ecs"] + eps
            )

        # Add DeltaTM (TM score relative to best possible template)
        for baseline in Config.BASELINES.values():
            bl_name = baseline.name
            for label, df in datasets:
                if label == "multimers" and baseline.mul_afa_file is None:
                    continue

                df[f"{bl_name.upper()} {label.upper()} ∆TM Score"] = (
                    df[f"{bl_name.upper()} {label.upper()} TM Score"]
                    - df[f"{bl_name.upper()} TM Homolog Score"]
                )

        # Cache the analyzed output
        mon_df.to_csv(Config.MONOMER_ANALYZED_CSV, index=False)
        mul_df.to_csv(Config.MULTIMER_ANALYZED_CSV, index=False)

    # Print summary
    if print_status:
        datasets = [("monomers", mon_df), ("multimers", mul_df)]
        for label, dataset in datasets:
            print(f"--------Statistics for {label}--------")
            for bl_name, bl in Config.BASELINES.items():
                is_multimer = label == "multimers"
                if is_multimer and bl.mul_afa_file is None:
                    continue

                bl_key = f"{bl_name.upper()} {label.upper()}"
                tm_col = f"{bl_key} TM Score"
                dtm_col = f"{bl_key} ∆TM Score"
                wc_col = f"{bl_key} INF_WC"
                nwc_col = f"{bl_key} INF_NWC"
                print(f"{bl_name}: ", end="")
                n_completed = len(dataset) - dataset[tm_col].isna().sum()
                print(f"{n_completed}/{len(dataset)} completed, ", end="")
                if n_completed != 0:
                    min_tm = dataset[tm_col].min()
                    max_tm = dataset[tm_col].max()
                    avg_tm = dataset[tm_col].mean(skipna=True)
                    delta_tm = dataset[dtm_col].mean(skipna=True)

                    # For INF WC/NWC, skip structures where INF WC/NWC could
                    # not be calculated (-1.0)
                    wc = dataset[dataset[wc_col] != -1.0]
                    nwc = dataset[dataset[nwc_col] != -1.0]
                    inf_wc = wc[wc_col].mean(skipna=True)
                    inf_nwc = nwc[nwc_col].mean(skipna=True)
                    print(
                        f"TM {min_tm:.3f}-{max_tm:.3f} (avg. {avg_tm:.3f}, "
                        f"∆={delta_tm:.3f}, WC={inf_wc:.3f}, NWC={inf_nwc:.3f})",
                        end="",
                    )
                print("")

    return mon_df, mul_df


def plot_tm_scores(
    baseline: Baseline = None,
    method: str = "spearman",
    x_col="{baseline} TM Homolog Score",
    plot_diag=True,
    filter_na=False,
):
    """
    Plots TM train for different baselines.

    Parameters
    ----------
    baseline : Optional[Baseline] (default = None)
        The baseline to measure.  If None, evaluates all baselines. Otherwise,
        evaluates only the specified baseline.
    method : str (default = "spearman")
        The method to use for correlation.  Defaults to "spearman".  Another
        common option would be "pearson".
    x_col : str (default = "{baseline} TM Homolog Score")
        The column to use for the x axis.  "{baseline}" will be filled with the
        uppercase baseline's name.  Alternatively, it can be left out to
        indicate that x_col is not baseline dependent.
    plot_diag : bool (default = True)
        If True, plots the diagonal line y=x.
    filter_na : bool (default = False)
        If True, filters rows where x_col is NA/None.
    """
    # Load both CSV files
    mon_df, mul_df = get_baseline_scores()

    if filter_na:
        mon_df = mon_df[mon_df[x_col].notna() & (mon_df[x_col] > 0.0)]
        mul_df = mul_df[mul_df[x_col].notna() & (mul_df[x_col] > 0.0)]
        print(f"{len(mon_df)} monomers after filtering...")
        print(f"{len(mul_df)} multimers after filtering...")

    # Plot and give correlations for all datasets
    baselines = Config.BASELINES.values() if baseline is None else [baseline]
    for baseline in baselines:
        # Create the plot (adjust figure size to match the snippet's style)
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot for monomer
        label = baseline.name.upper()
        x_col_bl = x_col.format(baseline=baseline.name.upper())
        y_mon_col = f"{label} MONOMERS TM Score"
        ax.scatter(
            mon_df[x_col_bl],
            mon_df[y_mon_col],
            label="Monomer",
            marker="o",
            # Use the dracula color for the fill, black edges to mirror the snippet's style
            facecolor=dracula_blue,
            edgecolor="black",
            s=80,  # Marker size
            linewidth=1,  # Edge line width
            alpha=1.0,
        )
        mon_spearman = mon_df[[x_col_bl, y_mon_col]].corr(method=method).iloc[0, 1]
        print(f"{label} - Monomer Spearman: {mon_spearman:.3f}")

        # Scatter plot for multimer
        if baseline.mul_afa_file is not None:
            y_mul_col = f"{label} MULTIMERS TM Score"
            ax.scatter(
                mul_df[x_col_bl],
                mul_df[y_mul_col],
                label="Multimer",
                marker="D",
                facecolor=dracula_orange,
                edgecolor="black",
                s=80,
                linewidth=1,
                alpha=1.0,
            )
            mul_spearman = mul_df[[x_col_bl, y_mul_col]].corr(method=method).iloc[0, 1]
            print(f"{label} - Multimer Spearman: {mul_spearman:.3f}")

        # Plot the line y=x in solid black
        if plot_diag:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            line_min = min(x_min, y_min)
            line_max = max(x_max, y_max)
            ax.plot(
                [line_min, line_max], [line_min, line_max], color="black", linestyle="-"
            )

            # Optionally adjust the axes to ensure the diagonal line spans the entire plot
            ax.set_xlim(line_min, line_max)
            ax.set_ylim(line_min, line_max)

        # Label axes using a larger font size (per snippet style)
        ax.set_xlabel(f"{x_col_bl}", fontsize=axis_label_font_size)
        ax.set_ylabel("Predicted TM score", fontsize=axis_label_font_size)

        # Move legend to the bottom-right
        ax.legend(loc="lower right")

        plt.tight_layout()
        out_fname = Path(f"figures/{baseline.name}_tm_score_vs_{x_col_bl}.png")
        out_fname.parent.mkdir(exist_ok=True)
        plt.savefig(out_fname, dpi=300)
        plt.close(fig)
