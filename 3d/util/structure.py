#!/usr/bin/env python3

###############################################################################
# `structure.py`: Helper classes for working with structures.
###############################################################################

from __future__ import annotations

import gzip
import os
import re
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import evcouplings.align.alignment as alignment
import gemmi
import requests
from Bio.Seq import Seq
from evcouplings.utils.system import ResourceError
from gemmi import Residue, ResidueSpan, Structure
from gemmi.cif import Document

from util import BA1_TO_BA2, ChainID, Config, ContactMap, EqClassID, Residues
from util.sequence import FamHits


class ResidueType(Enum):
    """
    Enum class for categorizing residues.
    """

    RNA = auto()
    DNA = auto()
    PROTEIN = auto()
    MOD_RNA = auto()
    MOD_DNA = auto()
    MOD_PROTEIN = auto()
    SOLVENT = auto()
    HETATM = auto()  # Everything else

    @staticmethod
    def get_residue_type(residue: Residue) -> ResidueType:
        """
        Returns the residue type of the input residue.

        Parameters:
            residue (pd.DataFrame): A residue acquired from an EVCouplings
             Chain object.
        """
        is_polymer = residue.entity_type == gemmi.EntityType.Polymer
        residue = residue.name
        residue_type: ResidueType

        # NOTE(MCA): Gemmi has its own implementation of residue/polymer
        #   typing, but it has limited awareness of modified protein/nucleic
        #   acid residues.
        if is_polymer and residue in Residues.RNA:
            residue_type = ResidueType.RNA
        elif is_polymer and residue in Residues.DNA:
            residue_type = ResidueType.DNA
        elif is_polymer and residue in Residues.Protein:
            residue_type = ResidueType.PROTEIN
        elif (
            is_polymer
            and residue in Residues.ModNA
            and not Residues.ModNA[residue].startswith("D")
        ):
            residue_type = ResidueType.MOD_RNA
        elif (
            is_polymer
            and residue in Residues.ModNA
            and Residues.ModNA[residue].startswith("D")
        ):
            residue_type = ResidueType.MOD_DNA
        elif is_polymer and residue in Residues.ModProtein:
            residue_type = ResidueType.MOD_PROTEIN
        elif (
            residue in Residues.Water
            or residue in Residues.SmallMolecule
            or residue in Residues.Element
        ):
            residue_type = ResidueType.SOLVENT
        else:
            residue_type = ResidueType.HETATM

        return residue_type

    @staticmethod
    def get_restype_lists(
        chain: ResidueSpan,
    ) -> DefaultDict[ResidueType, List[Residue]]:
        """
        Returns the residue types in a given chain along with a list of
        residues of each type.
        """
        restype_lists = defaultdict(list)
        for residue in chain:
            restype_lists[ResidueType.get_residue_type(residue)].append(residue)
        return restype_lists

    @staticmethod
    def is_mod_res_type(residue_type: ResidueType) -> bool:
        """
        Returns True if the input residue type is a modified residue type, else
        False.
        """
        if not hasattr(ResidueType.is_mod_res_type, "MODIFIED_RESIDUES"):
            ResidueType.is_mod_res_type.MODIFIED_RESIDUES = {
                ResidueType.MOD_RNA,
                ResidueType.MOD_DNA,
                ResidueType.MOD_PROTEIN,
            }
        return any(
            mod_residue_type == residue_type
            for mod_residue_type in ResidueType.is_mod_res_type.MODIFIED_RESIDUES
        )

    @staticmethod
    def to_chain_type(res_type: ResidueType) -> ChainType:
        return ResidueType._res_type_to_chain_type[res_type]


class ChainType(Enum):
    """
    Enum class for categorizing chains.
    """

    ANY = auto()  # Any chain type
    NA = auto()  # RNA, DNA, or NA_HYBRID
    RNA = auto()
    DNA = auto()
    PROTEIN = auto()
    NA_HYBRID = auto()  # Mixture of DNA and RNA
    HETATM = auto()
    SOLVENT = auto()
    UNKNOWN = auto()  # Everything else

    @staticmethod
    def get_chain_type(
        chain: ResidueSpan,
    ) -> (ChainType, DefaultDict[ChainType, List[Residue]]):
        """
        Returns the chain type of the input chain, along with the counts of
        each residue type (useful for hybrid chains).
        """
        restype_lists = ResidueType.get_restype_lists(chain)
        residue_types = set(restype_lists.keys())

        if not residue_types - {ResidueType.SOLVENT}:
            return ChainType.SOLVENT, restype_lists
        else:
            residue_types -= {ResidueType.SOLVENT}

        if not residue_types - {ResidueType.HETATM}:
            return ChainType.HETATM, restype_lists
        else:
            residue_types -= {ResidueType.HETATM}

        if not residue_types - {ResidueType.RNA, ResidueType.MOD_RNA}:
            return ChainType.RNA, restype_lists
        elif not residue_types - {ResidueType.DNA, ResidueType.MOD_DNA}:
            return ChainType.DNA, restype_lists
        elif not residue_types - {
            ResidueType.PROTEIN,
            ResidueType.MOD_PROTEIN,
        }:
            return ChainType.PROTEIN, restype_lists
        elif not residue_types - {
            ResidueType.RNA,
            ResidueType.DNA,
            ResidueType.MOD_RNA,
            ResidueType.MOD_DNA,
        }:
            return ChainType.NA_HYBRID, restype_lists

        return ChainType.UNKNOWN, restype_lists


ChainType.ALL_NAS = (ChainType.RNA, ChainType.DNA, ChainType.NA_HYBRID)

ResidueType._res_type_to_chain_type = {
    ResidueType.RNA: ChainType.RNA,
    ResidueType.DNA: ChainType.DNA,
    ResidueType.PROTEIN: ChainType.PROTEIN,
    ResidueType.MOD_RNA: ChainType.RNA,
    ResidueType.MOD_DNA: ChainType.DNA,
    ResidueType.MOD_PROTEIN: ChainType.PROTEIN,
    ResidueType.SOLVENT: ChainType.SOLVENT,
    ResidueType.HETATM: ChainType.HETATM,
}


class ChainInfo:
    """
    Classifies information about an input chain.

    Properties:
      src_organism (str): The organism from which this entity was obtained.  In
        order of descending priority, this will be the native organism, gene
        organism, or host organism if available.  Otherwise, it will be the
        empty string.

    Members:
        syn_organism (str):  Scientific name of the organism from
          which this entity was obtained (if synthetic).  Empty string if not supplied.
    """

    __asym_chain_id: ChainID  # The asym. chain ID of this chain
    __entity_id: int  # The entity ID of this chain
    __gene_organism: Optional[str]  # Gene source scientific name
    __host_organism: Optional[str]  # Host organism scientific name

    def __init__(
        self,
        asym_chain_id: ChainID,
        entity_id: int,
        gene_organism: Optional[str] = None,
        host_organism: Optional[str] = None,
        nat_organism: Optional[str] = None,
        syn_organism: Optional[str] = None,
    ):
        self.__asym_chain_id = asym_chain_id
        self.__entity_id = entity_id
        self.__gene_organism = gene_organism
        self.__host_organism = host_organism
        self.__nat_organism = nat_organism
        self.syn_organism = syn_organism if syn_organism is not None else ""

    @property
    def src_organism(self) -> str:
        """
        Retrieve the scientific name of the organism from which this entity
        was obtained.
        """
        return (
            self.__nat_organism
            if self.__nat_organism is not None
            else self.__gene_organism
            if self.__gene_organism is not None
            else self.__host_organism
            if self.__host_organism is not None
            else ""
        )


def strip_cif_value(cif_value: str) -> str:
    """
    Strips a CIF value of meaningless characters.
    """
    strip_chars = "\"\r\n';?"
    return cif_value.strip(strip_chars)


def get_cif_key(cif: Document, key: str, default="") -> str:
    """
    Returns a CIF key from the input Block.  Any list values are joined using
    ", " into a single string.  None and "?" values are converted to the empty
    string.
    """
    category, tag = key.split(".")
    table = cif.sole_block().find_mmcif_category(category)
    try:
        column = table.find_column(tag)
    except RuntimeError:
        column = []

    def is_none(value: str):
        return value == "" or value == "?" or value is None

    # Default to None or the first value if available
    value = None if len(column) == 0 else column[0]

    # If there are multiple values, merge them
    if len(column) > 1:
        if all(is_none(v) for v in column):
            value = None
        else:
            value = ", ".join(column)

    return strip_cif_value(value) if not is_none(value) else default


def escape_quotes(string: str):
    """
    Escapes the quotation marks in an input string for CSV.
    """
    return string.replace('"', '""').replace("'", "''")


def get_structure(pdb_id: str) -> Tuple[Document, Structure]:
    """
    Returns the selected PDB from RCSB as an EVCouplings PDB object.  PDB
    download results are cached for reuse in Config.OUT_PREFIX.

    Parameters:
        pdb_id (str): The PDB ID of interest.

    Returns:
        cif (Document):  A Document object representing of the full CIF file.
        structure (Structure):  A Structure object representing the assembly
          CIF file.
    """
    # Download the full CIF and assembly CIFs
    pdb_id = pdb_id.lower()
    out_prefix = Config.get_out_prefix(pdb_id)
    assembly_url = Config.RCSB_ASSEMBLY_URL.format(pdb_id=pdb_id)
    assembly_out = Config.RCSB_ASSEMBLY_FILE.format(pdb_id=pdb_id)
    full_url = Config.RCSB_FULL_URL.format(pdb_id=pdb_id)
    full_out = Config.RCSB_FULL_FILE.format(pdb_id=pdb_id)

    def fetch_cif(source_url: str, out_file: str):
        os.makedirs(out_prefix, exist_ok=True)
        if not os.path.exists(out_file) or os.stat(out_file).st_size == 0:
            with open(out_file, "w") as f:
                r = requests.get(source_url)
                f.write(gzip.decompress(r.content).decode("utf-8"))

    fetch_cif(assembly_url, assembly_out)
    fetch_cif(full_url, full_out)

    cif = gemmi.cif.read_file(full_out)
    structure = gemmi.read_structure(assembly_out)

    # Clean up the input structure
    structure.remove_alternative_conformations()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_empty_chains()

    return (cif, structure)


def iterate_contacts(
    grid: gemmi.NeighborSearch,
    model: gemmi.Model,
    asym_ids: Set[ChainID] = {},
    max_radius=5.0,
) -> Generator[Tuple[Residue, gemmi.CRA], None, None]:
    """
    Generates all contacts for every chain in the input model.

    Parameters
    ----------
    grid (gemmi.NeighborSearch):
        A populated gemmi.NeighborSearch object to use for contact searching.
    model (gemmi.Model):
        The model to search.
    asym_ids (Set[ChainID]):
        Specific asym. IDs to search for cofactor contacts.
    max_radius (float):
        The maximum distance in Å to consider for contacts.
    """
    for chain in model.subchains():
        chain_id = chain.subchain_id()
        if chain_id not in asym_ids:
            continue

        for residue in chain:
            for atom in residue:
                contacts = grid.find_neighbors(atom=atom, max_dist=max_radius)
                for hit in contacts:
                    yield (residue, hit.to_cra(model))


def get_self_contacts(
    grid: gemmi.NeighborSearch,
    model: gemmi.Model,
    chain: gemmi.ResidueSpan,
    max_radius=5.0,
    min_neighbor_dist=6,
) -> ContactMap:
    """
    Retrieves contacts between a chain and itself. The resulting dictionary maps
    each input chain residue to its self-contacting residues.

    Parameters:
        grid (gemmi.NeighborSearch):  A populated gemmi.NeighborSearch object
          to use for contact searching.
        model (gemmi.Model): The model to search.
        chain (gemmi.ResidueSpan): The chain to check for self contacts.
        max_radius (float):  The maximum distance in Å to consider for
          contacts.
    """
    self_contacts = {}
    for residue, hit in iterate_contacts(
        grid, model, {chain.subchain_id()}, max_radius
    ):
        r1 = residue
        r2 = hit.residue
        if (
            r2.subchain == r1.subchain
            and abs(r2.label_seq - r1.label_seq) >= min_neighbor_dist
        ):
            self_contacts.setdefault(r1, set()).add(r2)
            self_contacts.setdefault(r2, set()).add(r1)

    return self_contacts


def get_chain_coverages(
    grid: gemmi.NeighborSearch,
    model: gemmi.Model,
    asym_ids: Set[ChainID] = {},
    max_radius: float = 5.0,
    only_nucleic: bool = False,
) -> Dict[ChainID, float]:
    """
    Gets the % coverage of the specified RNA chains in the dataset.  Here %
    coverage means the % of the input chain's residues that are in contact with
    different polymer chain.

    grid (gemmi.NeighborSearch):
        A populated gemmi.NeighborSearch object to use for contact searching.
    model (gemmi.Model):
        The model to search.
    asym_ids (Set[ChainID]):
        Specific asym. IDs to search for cofactor contacts.
    max_radius (float):
        The maximum distance in Å to consider for contacts.
    only_nucleic (bool):
        If True, only looks for coverage by nucleic polymers.
    """
    residues_covered = {asym_id: set() for asym_id in asym_ids}
    chain_lengths = {}

    for chain in model.subchains():
        chain_id = chain.subchain_id()
        if chain_id in asym_ids and chain_id not in chain_lengths:
            chain_lengths[chain_id] = chain.length()

    for residue, hit in iterate_contacts(grid, model, asym_ids, max_radius):
        is_polymer = hit.residue.entity_type == gemmi.EntityType.Polymer
        is_nucleic = hit.residue.name in Residues.NA

        # interxtal contacts are contacts with a symmetry expanded copy of
        # itself.  These subchains come in the form "<ChainID>-<2,3,...,N>"
        # where N indicates the which symmetry copy the subchain belongs to.
        res_sc = residue.subchain.split("-")
        hit_sc = hit.residue.subchain.split("-")
        is_interchain = res_sc[0] != hit_sc[0]
        is_interxtal = not is_interchain and (
            len(res_sc) != len(hit_sc) or (len(res_sc) == 2 and res_sc[1] != hit_sc[1])
        )

        if (
            is_polymer
            and (is_interchain or is_interxtal)
            and (not only_nucleic or is_nucleic)
        ):
            residues_covered[residue.subchain].add(residue.label_seq)

    chain_coverages = {
        asym_id: len(residues_covered[asym_id]) / chain_lengths[asym_id]
        for asym_id in asym_ids
    }
    return chain_coverages


def get_interchain_contacts(
    grid: gemmi.NeighborSearch,
    model: gemmi.Model,
    asym_ids: Iterable[ChainID],
    max_radius=5.0,
    only_nucleic=False,
    only_cofactors=False,
) -> ContactMap:
    """
    Retrieves the cofactor contacts in the given structure for the given entity
    ID.  The resulting dictionary maps each polymer residue in the input
    asym. ID chains to its non-polymer residue contacts, and vice versa.

    Parameters:
        model (gemmi.Model): The model to search.
        asym_ids (Iterable[ChainID]): The asym. IDs to search for cofactor
          contacts.
        grid (gemmi.NeighborSearch):  A populated gemmi.NeighborSearch object
          to use for contact searching.
        max_radius (float):  The maximum distance in Å to consider for
          contacts.
        only_nucleic: If True, gets only NA interchain contacts.
        only_cofactors: If True, gets only interchain cofactor contacts.
    """
    if (only_nucleic + only_cofactors) != 1:
        raise ValueError("Must supply either only_nucleic or only_cofactors")
    if not isinstance(asym_ids, set):
        asym_ids = set(asym_id for asym_id in asym_ids)

    # Identify contacts involving chains listed in asym_ids
    neighbors: ContactMap = {}
    for residue, hit in iterate_contacts(grid, model, asym_ids, max_radius):
        is_polymer = hit.residue.entity_type == gemmi.EntityType.Polymer
        is_nucleic = hit.residue.name in Residues.NA
        is_interchain = hit.residue.subchain != residue.subchain

        if (only_cofactors and not is_polymer) or (
            only_nucleic and is_interchain and is_polymer and is_nucleic
        ):
            r1 = residue
            r2 = hit.residue
            neighbors.setdefault(r1, set()).add(r2)
            neighbors.setdefault(r2, set()).add(r1)

    return neighbors


class StructureInfo:
    """
    Classifies information about an input structure.

    Members:
        HEADERS:  Headers for the values output by `get_data()`.
        assembly (Structure):  The biological assembly as a Gemmi Structure.
        asym_id_to_auth_id (Dict[ChainID, ChainID]):  Mapping from each asym.
          ID to its respective auth. ID.
        auth_id_to_asym_ids (DefaultDict[ChainID, Set[ChainID]]):  Mapping
          from each auth. ID to its respective asym. ID(s).
        asym_id_to_entity_id (Dict[ChainID, EntityID]):  Mapping from each asym
          ID to its respective entity ID.
        cofactors (DefaultDict[ChainID, Set[ResName]]):  Mapping from chain ID
          to list of cofactors.  Cofactors are defined as non-polymer atoms
          within a cutoff distance (5.0 Å) of at least two distinct segments of
          a polymer entity.  A segment is defined as a region of +/- 3 residues
          from the contact point.
        cif (Document):  The full CIF as a Gemmi Document.
        chain_coverages (Dict[ChainType, Dict[ChainID, float]]):  Mapping from
          ChainType to mapping from ChainID to % of residues in contact with a
          a different polymer chain of that type.
        chain_infos (Dict[ChainID, ChainInfo]):  Mapping from ChainID to
          ChainInfo objects.
        chain_types(Dict[ChainID, ChainType]):  Mapping from each ChainID in
          the input structure to its respective type.
        chains_of_type (DefaultDict[ChainType, List[Chain]]):  Mapping from
          ChainType to a list of chains of that ChainType.
        eq_classes (Dict[ChainID, Tuple[EqClassID, int]]):  Mapping from auth.
          chain IDs to equivalence class IDs and the no. of chains each
          contains.
        fam_hits (Optional[Dict[ChainID, FamHits]]):  Mapping from ChainIDs to
          FamHits objects representing matching Pfam/Rfam families.
        keywords (List[str]):  Keywords for this structure.
        method (str):  The method for determining this structure.
        neighbors (DefaultDict[ChainID, Set[ChainID]]): Set of neighboring
          nucleic acid neighbor chains for each RNA chain in the input.
          Neighbors are defined as those with residues within 5 Å.
        published_date (str):  Date this PDB was published.
        residues (Dict[ResidueType, Set[ResName]]):  Mapping from ResidueType
          to residues of that type in the structure.  Useful for tracking,
          e.g., modified residues.
        resolution (str):  The reported resolution of this structure.
        revision_dates (List[str]):  Dates this PDB was revised.
        self_contacts (Dict[ChainID, ContactMap]):  Mapping from input chains
          to ContactMap objects describing self contacts between residues.
        sequences (Dict[ChainID, Seq]):  Mapping from each ChainID in the
          input structure to its respective sequence.
        sequences_unmod (Dict[ChainID, Seq]):  Same as `sequences` but
          with modified residues converted to their standard counterparts
          (e.g., 6MA -> A).
        sources (Set[str]): A list of strings describing the sources this
          structure info's PDB ID came from.
    """

    __slots__ = (
        "assembly",
        "asym_id_to_auth_id",
        "auth_id_to_asym_ids",
        "asym_id_to_entity_id",
        "cofactors",
        "chain_coverages",
        "chain_infos",
        "chain_types",
        "chains_of_type",
        "cif",
        "eq_classes",
        "fam_hits",
        "keywords",
        "method",
        "neighbors",
        "published_date",
        "residues",
        "resolution",
        "revision_dates",
        "self_contacts",
        "sequences",
        "sequences_unmod",
        "sources",
    )

    HEADERS = [
        "PDB ID",
        "Asym. Chain ID",
        "Auth. Chain ID",
        "Sequence Cluster",
        "Rfam Cluster",
        "Source(s)",
        "Name",
        "Published",
        "Keywords",
        "Method",
        "Resolution",
        "Organism",
        "Synthetic organism",
        # equivalence class ID from RNA 3D Hub (R3DH)
        "EC ID",
        # no. of chains in R3DH integrated functional element (IFE)
        "IFE chains",
        "Self Structured",
        "% covered (any polymer)",
        "% covered (only NA)",
        "# of neighbor NA chains",
        "# of RNA chains",
        "# of protein chains",
        "# of DNA chains",
        "# of hybrid chains",
        "# of HETATM chains",
        "# of solvent chains",
        "# of unknown chains",
        "N_nt",  # number of nucleotids
        "N_aa",  # number of amino acids
        "N",  # N_nt + N_aa
        "Solvent residues",
        "Hetatm residues",
        "Mod. RNA residues",
        "Mod. DNA residues",
        "Mod. protein residues",
        "Cofactors",
        "Sequence",
        "Sequence (unmod.)",
        "L",
        "Fraction missing",
        "Rfam",
        "Rfam N/L",
        "Rfam L",
        "Rfam fraction observed",
        "Rfam bitscore",
        "Rfam E-value",
    ]

    # Order must match HEADERS
    __tracked_chain_types = [
        ChainType.RNA,
        ChainType.PROTEIN,
        ChainType.DNA,
        ChainType.NA_HYBRID,
        ChainType.HETATM,
        ChainType.SOLVENT,
        ChainType.UNKNOWN,
    ]

    # Order must match HEADERS
    __tracked_residue_types = [
        ResidueType.SOLVENT,
        ResidueType.HETATM,
        ResidueType.MOD_RNA,
        ResidueType.MOD_DNA,
        ResidueType.MOD_PROTEIN,
    ]

    # Regex for identifying modified residues in annotated sequences
    __mod_residue_re = re.compile(r"-\(([A-Za-z0-9_-]*)\)-")

    def __init__(
        self,
        cif: Document,
        assembly: Structure,
        eq_classes: Dict[ChainID, Tuple[EqClassID, int]],
        sources: Set[str],
    ):
        self.cif = cif
        self.assembly = assembly
        self.eq_classes = eq_classes
        self.sources = sources

        # Identify which chain IDs to process
        model = self.assembly[0]
        chains = model.subchains()

        # Process the iterable of Chain IDs
        self.residues = defaultdict(set)
        self.chain_types = {}
        self.chains_of_type = defaultdict(list)
        for chain in chains:
            chain_id = chain.subchain_id()

            # Skip symmetry expanded copies of chains (coming with names like
            # "B-2")
            if len(chain_id.split("-")) > 1:
                continue

            # Mark NA_HYBRIDs as the dominant NA type if they exceed
            # HYBRID_CUTOFF
            chain_type, restype_lists = ChainType.get_chain_type(chain)
            if chain_type == ChainType.NA_HYBRID:
                dominant_chain_type = next(
                    (
                        ResidueType.to_chain_type(rtype)
                        for rtype, residues in restype_lists.items()
                        if len(residues) / chain.length() > Config.HYBRID_CUTOFF
                    ),
                    None,
                )
                chain_type = dominant_chain_type or chain_type

            for res_type, residues in restype_lists.items():
                if res_type in self.__tracked_residue_types:
                    for residue in residues:
                        self.residues[res_type].add(residue.name)

            self.chain_types[chain_id] = chain_type
            self.chains_of_type[chain_type].append(chain)

        # --- Map from asym. ID to sequence ---
        block = self.cif.sole_block()
        poly_seq_table = block.find_mmcif_category("_pdbx_poly_seq_scheme")
        asym_ids = poly_seq_table.find_column("asym_id")
        mon_ids = poly_seq_table.find_column("mon_id")
        rna_chains = self.chains_of_type[ChainType.RNA]
        polymer_chains = (
            rna_chains
            + self.chains_of_type[ChainType.DNA]
            + self.chains_of_type[ChainType.PROTEIN]
            + self.chains_of_type[ChainType.NA_HYBRID]
        )
        rna_asym_ids = set(chain.subchain_id() for chain in rna_chains)
        protein_asym_ids = {
            chain.subchain_id() for chain in self.chains_of_type[ChainType.PROTEIN]
        }
        polymer_asym_ids = set(chain.subchain_id() for chain in polymer_chains)
        self.sequences = {}
        self.sequences_unmod = {}
        for asym_id, mon_id in zip(asym_ids, mon_ids):
            if asym_id in polymer_asym_ids:
                self.sequences.setdefault(asym_id, "")
                self.sequences[asym_id] += (
                    mon_id if len(mon_id) == 1 else f"-({mon_id})-"
                )

        # Convert to Seq objects
        self.sequences_unmod = {
            asym_id: Seq(
                str(StructureInfo.__unmodify_seq(seq)).replace(
                    "?", "X" if asym_id in protein_asym_ids else "N"
                )
            )
            for asym_id, seq in self.sequences.items()
        }
        self.sequences = {asym_id: Seq(seq) for asym_id, seq in self.sequences.items()}

        # --- Map from asym. ID to entity ID ---
        self.asym_id_to_entity_id = {}
        atom_site = block.find_mmcif_category("_atom_site")
        asym_ids = atom_site.find_column("label_asym_id")
        entity_ids = atom_site.find_column("label_entity_id")
        for asym_id, entity_id in zip(asym_ids, entity_ids):
            self.asym_id_to_entity_id[asym_id] = entity_id

        # --- Map from auth ID to asym. ID and vice versa ---
        # NOTE(MCA):  The relationship of asym. ID to auth. ID is either
        #   one-to-one or many-to-one.  There may be multiple asym. IDs for any
        #   given auth ID.
        self.auth_id_to_asym_ids = defaultdict(set)
        self.asym_id_to_auth_id = {}
        auth_ids = atom_site.find_column("auth_asym_id")
        for asym_id, auth_id in zip(asym_ids, auth_ids):
            self.auth_id_to_asym_ids[auth_id].add(asym_id)
            self.asym_id_to_auth_id[asym_id] = auth_id

        # --- Identify fam hits for RNAs ---
        self.fam_hits = {}
        if len(rna_chains) <= Config.MAX_RFAM_MSAS:
            for chain in rna_chains:
                chain_id = chain.subchain_id()
                sequence = self.sequences_unmod[chain_id]
                auth_chain_id = self.asym_id_to_auth_id[chain_id]

                if Config.RNA_MIN_NT <= len(sequence) <= Config.RNA_MAX_NT:
                    with NamedTemporaryFile(mode="w+", delete=True) as tmp:
                        alignment.write_fasta(((chain_id, str(sequence)),), tmp)
                        tmp.seek(0)
                        self.fam_hits[chain_id] = FamHits.from_fam(
                            self.pdb_id,
                            chain_id,
                            auth_chain_id,
                            tmp,
                            FamHits.Source.RFAM,
                        )
                else:
                    self.fam_hits[chain_id] = None

        # --- Structural details ---
        self.revision_dates = get_cif_key(
            self.cif, "_pdbx_audit_revision_history.revision_date"
        )
        self.published_date = self.revision_dates.split(", ")[0]
        self.keywords = escape_quotes(get_cif_key(self.cif, "_struct_keywords.text"))
        self.method = get_cif_key(self.cif, "_exptl.method")
        if "X-RAY DIFFRACTION" in self.method:
            self.resolution = get_cif_key(
                self.cif, "_refine.ls_d_res_high", default="n.s."
            )
        elif "ELECTRON MICROSCOPY" in self.method:
            self.resolution = get_cif_key(
                self.cif, "_em_3d_reconstruction.resolution", default="n.s."
            )
        else:
            self.resolution = "N/A"

        # --- Organism details ---
        # NOTE(MCA): RCSB guarantees that the entity IDs in the biological
        #   assembly will match those in the full PDB.
        src_gen = block.find_mmcif_category("_entity_src_gen")
        if len(src_gen) > 0:
            gen_entity_ids = list(src_gen.find_column("entity_id"))
            gene_orgs = list(src_gen.find_column("pdbx_gene_src_scientific_name"))
            host_orgs = list(src_gen.find_column("pdbx_host_org_scientific_name"))
            eid_to_gene_org = {
                eid: strip_cif_value(org) for eid, org in zip(gen_entity_ids, gene_orgs)
            }
            eid_to_host_org = {
                eid: strip_cif_value(org) for eid, org in zip(gen_entity_ids, host_orgs)
            }
        else:
            eid_to_gene_org, eid_to_host_org = ({}, {})

        src_nat = block.find_mmcif_category("_entity_src_nat")
        if len(src_nat) > 0:
            nat_entity_ids = list(src_nat.find_column("entity_id"))
            nat_orgs = list(src_nat.find_column("pdbx_organism_scientific"))
            eid_to_nat_org = {
                eid: strip_cif_value(org) for eid, org in zip(nat_entity_ids, nat_orgs)
            }
        else:
            eid_to_nat_org = {}

        src_syn = block.find_mmcif_category("_pdbx_entity_src_syn")
        if len(src_syn) > 0:
            syn_entity_ids = list(src_syn.find_column("entity_id"))
            syn_orgs = list(src_syn.find_column("organism_scientific"))
            eid_to_syn_org = {
                eid: strip_cif_value(org) for eid, org in zip(syn_entity_ids, syn_orgs)
            }
        else:
            eid_to_syn_org = {}

        # --- Chain-specific structural details ---
        # Generate the ChainInfo objects for all RNA chains
        self.chain_infos = {}
        for chain in rna_chains:
            chain_id = chain.subchain_id()
            # Chains of the form '<id>-<#>' may appear for identical copies
            # of a chain in the biological assembly.
            chain_id = chain_id.split("-")[0]
            entity_id = self.asym_id_to_entity_id[chain_id]
            self.chain_infos[chain_id] = ChainInfo(
                asym_chain_id=chain_id,
                entity_id=entity_id,
                gene_organism=eid_to_gene_org.get(entity_id, None),
                host_organism=eid_to_host_org.get(entity_id, None),
                nat_organism=eid_to_nat_org.get(entity_id, None),
                syn_organism=eid_to_syn_org.get(entity_id, None),
            )

        # --- Identify RNA chain cofactors ---
        # Populate a grid-based neighbor search
        grid = gemmi.NeighborSearch(model, self.assembly.cell, max_radius=5.0).populate(
            include_h=False
        )
        cofactor_contacts = get_interchain_contacts(
            grid, model, rna_asym_ids, only_cofactors=True
        )

        # Identify cofactors (start -> cofactor -> end)
        self.cofactors = defaultdict(set)
        for start, neighbors in cofactor_contacts.items():
            if start.entity_type != gemmi.EntityType.Polymer:
                continue

            # Since we started with a polymer, all the neighbors must be
            # non-polymer
            for cofactor in neighbors:
                ends = cofactor_contacts[cofactor]

                # Start and end must be in the same entity, but at distinct
                # segments (> +/-3 residues from the original residue)
                for end in ends:
                    if (
                        end.entity_id == start.entity_id
                        and abs(end.seqid.num - start.seqid.num) > 3
                    ):
                        asym_chain_id = start.subchain
                        self.cofactors[asym_chain_id].add(cofactor.name)

        # --- Identify inter-chain nucleic acid contacts ---
        nucleic_contacts = get_interchain_contacts(
            grid, model, rna_asym_ids, only_nucleic=True
        )

        self.neighbors = defaultdict(set)
        for start, neighbors in nucleic_contacts.items():
            for neighbor in neighbors:
                self.neighbors[start.subchain].add(neighbor.subchain)

        # --- Write minimal chain PDBs for structural alignments ---
        for chain in rna_chains:
            chain_id = chain.subchain_id()
            pdb_out = Path(
                Config.CHAIN_MINIMAL_PDB_FILE.format(
                    pdb_id=self.pdb_id, chain_id=chain_id
                )
            )
            if not pdb_out.exists() or pdb_out.stat().st_size == 0:
                pdb_out.parent.mkdir(parents=True, exist_ok=True)
                new_structure = gemmi.Structure()
                new_model = gemmi.Model(0)
                new_chain = gemmi.Chain("A")

                # Add residues, using seqids as resids to preserve information
                # about each residue's relative location in the sequence.
                for residue in chain:
                    new_chain.add_residue(residue)
                    new_chain[-1].seqid = gemmi.SeqId(f"{residue.label_seq}")

                new_model.add_chain(new_chain)
                new_structure.add_model(new_model)
                new_structure.write_minimal_pdb(str(pdb_out))

        # --- Get information about self-contacts ---
        self.self_contacts = {}
        for chain in rna_chains:
            chain_id = chain.subchain_id()
            self.self_contacts[chain_id] = get_self_contacts(
                grid,
                model,
                chain,
                min_neighbor_dist=Config.SELF_CONTACT_MIN_NEIGHBOR_DIST,
            )

        # --- Populate chain coverages ---
        self.chain_coverages = {
            ChainType.ANY: get_chain_coverages(
                grid, model, rna_asym_ids, Config.COVERAGE_RADIUS
            ),
            ChainType.NA: get_chain_coverages(
                grid, model, rna_asym_ids, Config.COVERAGE_RADIUS, only_nucleic=True
            ),
        }

    @staticmethod
    def from_pdb_id(
        pdb_id: str,
        sources: Set[str],
        eq_classes: Dict[ChainID, Tuple[EqClassID, int]] = None,
    ) -> StructureInfo:
        """
        Factory function for initializing a StructureInfo from a PDB ID.

        Parameters:
            pdb_id (str): The 4-letter PDB ID to check on RCSB.
            sources (List[str]): A list of strings describing the sources this
              PDB ID belongs to.
            eq_classes (Dict[ChainID, Tuple[EqClassID, int]]):
              Mapping from chain IDs to equivalence class IDs and the no. of
              chains each contains.
        """
        print(f"Processing {pdb_id}...")

        # Retrieve the full structure to identify resolution and keywords, as
        # well as the assembly for annotation.
        try:
            cif, assembly = get_structure(pdb_id)
        except (
            ResourceError,
            gzip.BadGzipFile,
            requests.exceptions.RequestException,
        ) as e:
            print(f"Unable to download {pdb_id}: {e}")
            return None
        except ValueError as e:
            print(f"Invalid PDB file for {pdb_id}: {e}")
            print("Try resetting the cache and trying again?")
            raise e

        return StructureInfo(
            cif, assembly=assembly, eq_classes=eq_classes, sources=sources
        )

    @staticmethod
    def __unmodify_res(match: re.Match[str]):
        r"""
        Returns the one-letter code for a mod. residue or "?" if a suitable
        replacement is not known.

        Parameters:
            match (re.Match[str]): A regex match of the form r"-\((.*)\)-"
              where the inner capturing group is the modified residue name of
              interest.
        """
        mod_resname = match.group(1)
        one_letter_resname = (
            Residues.ModNA.get(mod_resname)
            or Residues.ModProtein.get(mod_resname)
            or Residues.ProteinTo1Letter.get(mod_resname)
            or Residues.DNATo1Letter.get(mod_resname)
            or "?"
        )

        return one_letter_resname

    @staticmethod
    def __unmodify_seq(sequence: str) -> str:
        """
        Returns the "unmodified" version of the input sequence.  This entails
        replacing any modified DNA, RNA, or protein residues with their
        unmodified counterparts.
        """
        return re.sub(
            StructureInfo.__mod_residue_re,
            StructureInfo.__unmodify_res,
            sequence,
        )

    def get_frac_missing_residues(self, chain_id: ChainID) -> int:
        """
        Retrieves the fraction of missing residues in the input chain ID.
        """
        n_seq_residues = len(self.sequences_unmod[chain_id])
        n_residues = self.assembly[0].get_subchain(chain_id).length()

        return (n_seq_residues - n_residues) / n_seq_residues

    @property
    def rna_chain_ids(self) -> Generator[ChainID, None, None]:
        """
        Returns a generator of the asym. chain IDs of RNAs in this structure.
        RNAs will only be yielded if they meet the length criterias specified
        in `Config`.
        """
        for chain in self.chains_of_type[ChainType.RNA]:
            chain_id = chain.subchain_id()
            if "-" in chain_id:
                continue
            sequence = self.sequences_unmod[chain_id]
            if Config.RNA_MIN_NT <= len(sequence) <= Config.RNA_MAX_NT:
                yield chain_id

    @property
    def pdb_id(self) -> str:
        return get_cif_key(self.cif, "_entry.id").lower()

    @property
    def pdb_name(self) -> str:
        return get_cif_key(self.cif, "_struct.title")

    def get_data(self, chain_id: ChainID = None) -> Union[List[str], List[List[str]]]:
        """
        Returns a list of strings representing the data stored in this
        StructureInfo object sorted according to HEADERS.  If chain_id is not
        provided, defaults to retrieving info on all RNA chains.

        Parameters:
            chain_id (ChainID): The asym. chain ID to get data for.
        """
        # Default to returning info on all chains
        if chain_id is None:
            return [self.get_data(chain_id) for chain_id in self.rna_chain_ids]

        # Otherwise, retrieve chain-specific information
        data = []

        # Structure information
        auth_id = self.asym_id_to_auth_id[chain_id]
        chain_key = f"{self.pdb_id.lower()}_{auth_id}"
        ba2_id = (
            auth_id
            if self.pdb_id.upper() not in BA1_TO_BA2
            else BA1_TO_BA2.get(auth_id, auth_id)
        )
        ba2_key = f"{self.pdb_id.lower()}_{ba2_id}"

        data.append(f'"{self.pdb_id.upper()}"')
        data.append(chain_id)
        data.append(auth_id)
        data.append(Config.SEQ_CLUST_REPR_CHAINS.get(chain_key, ""))
        data.append(str(Config.STRUCT_CLUST_COMPONENTS.get(chain_key, "")))
        if ba2_key in Config.RNA3DBENCH_DS3_CHAINS:
            self.sources.add("3D Bench DS3")
        if ba2_key in Config.RNA3DBENCH_DS4_CHAINS:
            self.sources.add("3D Bench DS4")
        data.append(f'"{", ".join(self.sources)}"')
        data.append(f'"{escape_quotes(self.pdb_name)}"')
        data.append(f'"{self.published_date}"')
        data.append(f'"{self.keywords}"')
        data.append(f'"{self.method}"')
        data.append(f'"{self.resolution}"')

        # Chain-specific structure information
        chain_info = self.chain_infos[chain_id]
        data.append(f'"{escape_quotes(chain_info.src_organism)}"')
        data.append(f'"{escape_quotes(chain_info.syn_organism)}"')

        # Equivalence class info
        eq_class, ife_size = self.eq_classes.get(auth_id, ("", ""))
        data.append(eq_class)
        data.append(ife_size)

        # Whether the RNA has structure (monomer or multimer)
        data.append(
            str(
                len(self.self_contacts[chain_id].keys())
                > Config.MIN_STRUCTURED_CONTACTS
            )
        )
        data.append(f"{self.chain_coverages[ChainType.ANY][chain_id]:.4f}")
        data.append(f"{self.chain_coverages[ChainType.NA][chain_id]:.4f}")
        n_na_neighbors = str(len(self.neighbors[chain_id]))
        data.append(n_na_neighbors)

        # Chain types
        for chain_type in self.__tracked_chain_types:
            data.append(str(len(self.chains_of_type[chain_type])))

        # Number of nucleotides/amino acids
        n_nt = sum(
            len(self.sequences_unmod[chain.subchain_id().split("-")[0]])
            for chain_type in ChainType.ALL_NAS
            for chain in self.chains_of_type[chain_type]
        )
        n_aa = sum(
            len(self.sequences_unmod[chain.subchain_id().split("-")[0]])
            for chain in self.chains_of_type[ChainType.PROTEIN]
        )
        n = n_nt + n_aa
        data.append(str(n_nt))
        data.append(str(n_aa))
        data.append(str(n))

        # Residue types
        for residue_type in self.__tracked_residue_types:
            data.append(f'"{", ".join(sorted(self.residues[residue_type]))}"')

        # Cofactors
        data.append(f'"{", ".join(sorted(self.cofactors[chain_id]))}"')

        # Sequence information
        if chain_id not in self.sequences:
            raise ValueError("Chain {chain_id} has no sequence information")

        seq = str(self.sequences[chain_id])
        seq_unmod = str(self.sequences_unmod[chain_id])
        data.append(seq)
        data.append(seq_unmod)
        data.append(str(len(seq_unmod)))

        # RNA missing residue %
        data.append(f"{self.get_frac_missing_residues(chain_id):.3f}")

        # Fam Hits data
        if (
            not self.fam_hits
            or chain_id not in self.fam_hits
            or not self.fam_hits[chain_id]
            or not self.fam_hits[chain_id][0]
        ):
            data += ["", "", "", "", ""]
        else:
            best_fam_hit = self.fam_hits[chain_id][0]
            data.append(best_fam_hit.name)
            data.append(f"{best_fam_hit.n_over_l:.3f}")
            data.append(str(best_fam_hit.model_len))
            data.append(
                f"{min(1.0, best_fam_hit.model_len / best_fam_hit.seq_len):.3f}"
            )
            data.append(f"{best_fam_hit.score:.3f}")
            data.append(f"{best_fam_hit.e_value:.3e}")

        return data
