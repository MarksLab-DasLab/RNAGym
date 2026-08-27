#!/usr/bin/env python3

###############################################################################
# `structure.py`: Helper classes for working with structures
###############################################################################

from __future__ import annotations

import gzip
import re
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    DefaultDict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import evcouplings.align.alignment as alignment
import gemmi
import requests
from Bio.Seq import Seq
from evcouplings.utils.system import ResourceError
from gemmi import Residue, ResidueSpan, Structure
from gemmi.cif import Document

from rnagym.config import Config3D
from rnagym.s3d.util import ChainID, Residues
from rnagym.s3d.util.sequence import FamHits


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
        #   acid residues
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


def parse_resolution(value: str) -> float | None:
    """Return the worst reported numeric resolution, if available."""
    resolutions = []
    for item in value.split(","):
        try:
            resolutions.append(float(item))
        except ValueError:
            pass
    return max(resolutions, default=None)


def canonicalize_sequence(sequence: str, protein: bool) -> str:
    """Replace unresolved polymer residues with the appropriate unknown token."""
    if protein:
        return sequence.replace("?", "X")
    sequence = sequence.upper().replace("T", "U")
    return "".join(base if base in "ACGU" else "N" for base in sequence)


def get_structure(pdb_id: str) -> Tuple[Document, Structure]:
    """
    Returns the selected PDB from RCSB as an EVCouplings PDB object.  PDB
    download results are cached under the shared 3D curation cache.

    Parameters:
        pdb_id (str): The PDB ID of interest.

    Returns:
        cif (Document):  A Document object representing of the full CIF file.
        structure (Structure):  A Structure object representing the assembly
          CIF file.
    """
    # Download the full CIF and assembly CIFs
    pdb_id = pdb_id.lower()
    out_prefix = Config3D.pdb_dir(pdb_id)
    assembly_url = Config3D.RCSB_ASSEMBLY_URL.format(pdb_id=pdb_id)
    assembly_out = Config3D.assembly_file(pdb_id)
    full_url = Config3D.RCSB_FULL_URL.format(pdb_id=pdb_id)
    full_out = Config3D.full_file(pdb_id)

    def fetch_cif(source_url: str, out_file: Path):
        out_prefix.mkdir(parents=True, exist_ok=True)
        if not out_file.is_file() or not out_file.stat().st_size:
            response = requests.get(source_url, timeout=60)
            response.raise_for_status()
            temporary = out_file.with_name(f".{out_file.name}.tmp")
            temporary.write_bytes(gzip.decompress(response.content))
            temporary.replace(out_file)

    fetch_cif(full_url, full_out)
    try:
        fetch_cif(assembly_url, assembly_out)
    except requests.exceptions.HTTPError as error:
        if error.response.status_code != 404:
            raise
        # Some entries have no deposited biological assembly
        assembly_out = full_out

    cif = gemmi.cif.read_file(str(full_out))
    structure = gemmi.read_structure(str(assembly_out))

    # Clean up the input structure
    structure.remove_alternative_conformations()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_empty_chains()

    return (cif, structure)


class ContactAnnotations(NamedTuple):
    """Summarize contacts involving the selected RNA chains."""

    cofactors: dict[ChainID, set[str]]
    neighbors: dict[ChainID, set[ChainID]]
    self_contacts: dict[ChainID, set[int]]
    polymer_coverage: dict[ChainID, float]
    nucleic_acid_coverage: dict[ChainID, float]


def get_contact_annotations(
    grid: gemmi.NeighborSearch,
    model: gemmi.Model,
    asym_ids: set[ChainID],
    max_radius: float = 5.0,
    min_neighbor_distance: int = 6,
) -> ContactAnnotations:
    """Collect every RNA contact annotation in one neighbor traversal."""
    chain_lengths = {}
    cofactor_residues = defaultdict(set)
    neighbors = defaultdict(set)
    nucleic_acid_covered = defaultdict(set)
    polymer_covered = defaultdict(set)
    self_contacts = defaultdict(set)

    for chain in model.subchains():
        chain_id = chain.subchain_id()
        if chain_id not in asym_ids:
            continue
        chain_lengths.setdefault(chain_id, chain.length())

        for residue in chain:
            for atom in residue:
                for hit in grid.find_neighbors(atom=atom, max_dist=max_radius):
                    neighbor = hit.to_cra(model).residue
                    if neighbor.entity_type != gemmi.EntityType.Polymer:
                        cofactor_residues[neighbor].add(residue)
                        continue

                    is_nucleic = neighbor.name in Residues.NA
                    if neighbor.subchain != residue.subchain and is_nucleic:
                        neighbors[residue.subchain].add(neighbor.subchain)
                    if neighbor.subchain == residue.subchain:
                        if (
                            abs(neighbor.label_seq - residue.label_seq)
                            >= min_neighbor_distance
                        ):
                            self_contacts[residue.subchain].update(
                                (residue.label_seq, neighbor.label_seq)
                            )
                        continue

                    # Symmetry-expanded copies share the base chain ID
                    source = residue.subchain.split("-")
                    target = neighbor.subchain.split("-")
                    is_interchain = source[0] != target[0]
                    is_intercrystal = not is_interchain and (
                        len(source) != len(target)
                        or (len(source) == 2 and source[1] != target[1])
                    )
                    if is_interchain or is_intercrystal:
                        polymer_covered[residue.subchain].add(residue.label_seq)
                        if is_nucleic:
                            nucleic_acid_covered[residue.subchain].add(
                                residue.label_seq
                            )

    cofactors = defaultdict(set)
    for cofactor, residues in cofactor_residues.items():
        for residue in residues:
            # Cofactors bridge distinct segments of the same polymer entity
            if any(
                other.entity_id == residue.entity_id
                and abs(other.seqid.num - residue.seqid.num) > 3
                for other in residues
            ):
                cofactors[residue.subchain].add(cofactor.name)

    return ContactAnnotations(
        cofactors,
        neighbors,
        self_contacts,
        {
            chain_id: len(polymer_covered[chain_id]) / length
            for chain_id, length in chain_lengths.items()
        },
        {
            chain_id: len(nucleic_acid_covered[chain_id]) / length
            for chain_id, length in chain_lengths.items()
        },
    )


class StructureInfo:
    """
    Classifies information about an input structure.

    Members:
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
        resolution (float | None):  The reported resolution of this structure.
        revision_dates (List[str]):  Dates this PDB was revised.
        self_contacts (Dict[ChainID, Set[int]]):  Residues in each chain with
          nonlocal self contacts.
        sequences (Dict[ChainID, Seq]):  Mapping from each ChainID in the
          input structure to its respective sequence.
        sequences_unmod (Dict[ChainID, Seq]):  Same as `sequences` but
          with modified residues converted to their standard counterparts
          (e.g., 6MA -> A).
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
    )

    __tracked_chain_types = {
        "num_dna_chains": ChainType.DNA,
        "num_hetatm_chains": ChainType.HETATM,
        "num_hybrid_chains": ChainType.NA_HYBRID,
        "num_protein_chains": ChainType.PROTEIN,
        "num_rna_chains": ChainType.RNA,
        "num_solvent_chains": ChainType.SOLVENT,
        "num_unknown_chains": ChainType.UNKNOWN,
    }

    __tracked_residue_types = {
        "hetatm_residues": ResidueType.HETATM,
        "modified_dna_residues": ResidueType.MOD_DNA,
        "modified_protein_residues": ResidueType.MOD_PROTEIN,
        "modified_rna_residues": ResidueType.MOD_RNA,
        "solvent_residues": ResidueType.SOLVENT,
    }

    # Regex for identifying modified residues in annotated sequences
    __mod_residue_re = re.compile(r"-\(([A-Za-z0-9_-]*)\)-")

    def __init__(self, cif: Document, assembly: Structure):
        self.cif = cif
        self.assembly = assembly

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
                        if len(residues) / chain.length() > Config3D.HYBRID_CUTOFF
                    ),
                    None,
                )
                chain_type = dominant_chain_type or chain_type

            for res_type, residues in restype_lists.items():
                if res_type in self.__tracked_residue_types.values():
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
                canonicalize_sequence(
                    str(StructureInfo.__unmodify_seq(seq)),
                    protein=asym_id in protein_asym_ids,
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
        #   given auth ID
        self.auth_id_to_asym_ids = defaultdict(set)
        self.asym_id_to_auth_id = {}
        auth_ids = atom_site.find_column("auth_asym_id")
        for asym_id, auth_id in zip(asym_ids, auth_ids):
            self.auth_id_to_asym_ids[auth_id].add(asym_id)
            self.asym_id_to_auth_id[asym_id] = auth_id

        # --- Identify fam hits for RNAs ---
        self.fam_hits = {}
        for chain in rna_chains:
            chain_id = chain.subchain_id()
            sequence = self.sequences_unmod[chain_id]
            auth_chain_id = self.asym_id_to_auth_id[chain_id]

            if len(sequence) >= Config3D.MIN_ANNOTATION_LENGTH:
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
        self.keywords = get_cif_key(self.cif, "_struct_keywords.text")
        self.method = get_cif_key(self.cif, "_exptl.method")
        if "X-RAY DIFFRACTION" in self.method:
            self.resolution = parse_resolution(
                get_cif_key(self.cif, "_refine.ls_d_res_high")
            )
        elif "ELECTRON MICROSCOPY" in self.method:
            self.resolution = parse_resolution(
                get_cif_key(self.cif, "_em_3d_reconstruction.resolution")
            )
        else:
            self.resolution = None

        # --- Organism details ---
        # NOTE(MCA): RCSB guarantees that the entity IDs in the biological
        #   assembly will match those in the full PDB
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
            # of a chain in the biological assembly
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

        # --- Identify RNA contacts ---
        grid = gemmi.NeighborSearch(model, self.assembly.cell, max_radius=5.0).populate(
            include_h=False
        )
        contacts = get_contact_annotations(
            grid,
            model,
            rna_asym_ids,
            Config3D.COVERAGE_RADIUS,
            Config3D.SELF_CONTACT_MIN_NEIGHBOR_DISTANCE,
        )
        self.cofactors = contacts.cofactors
        self.neighbors = contacts.neighbors
        self.self_contacts = contacts.self_contacts
        self.chain_coverages = {
            ChainType.ANY: contacts.polymer_coverage,
            ChainType.NA: contacts.nucleic_acid_coverage,
        }

        # --- Write minimal chain PDBs for structural alignments ---
        for chain in rna_chains:
            chain_id = chain.subchain_id()
            pdb_out = Config3D.chain_file(self.pdb_id, chain_id)
            if not pdb_out.exists() or pdb_out.stat().st_size == 0:
                pdb_out.parent.mkdir(parents=True, exist_ok=True)
                new_structure = gemmi.Structure()
                new_model = gemmi.Model(0)
                new_chain = gemmi.Chain("A")

                # Add residues, using seqids as resids to preserve information
                # about each residue's relative location in the sequence
                for residue in chain:
                    new_chain.add_residue(residue)
                    new_chain[-1].seqid = gemmi.SeqId(f"{residue.label_seq}")

                new_model.add_chain(new_chain)
                new_structure.add_model(new_model)
                new_structure.write_minimal_pdb(str(pdb_out))

    @staticmethod
    def from_pdb_id(pdb_id: str) -> Optional[StructureInfo]:
        """
        Factory function for initializing a StructureInfo from a PDB ID.

        Parameters:
            pdb_id (str): The 4-letter PDB ID to check on RCSB.
        """
        # Retrieve the full structure to identify resolution and keywords, as
        # well as the assembly for annotation
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

        return StructureInfo(cif, assembly=assembly)

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

    def get_frac_missing_residues(self, chain_id: ChainID) -> float:
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
        RNAs will only be yielded if they meet the length criteria specified
        in `Config3D`.
        """
        for chain in self.chains_of_type[ChainType.RNA]:
            chain_id = chain.subchain_id()
            if "-" in chain_id:
                continue
            sequence = self.sequences_unmod[chain_id]
            if len(sequence) >= Config3D.MIN_ANNOTATION_LENGTH:
                yield chain_id

    @property
    def pdb_id(self) -> str:
        return get_cif_key(self.cif, "_entry.id").lower()

    @property
    def pdb_name(self) -> str:
        return get_cif_key(self.cif, "_struct.title")

    def get_data(
        self, chain_id: ChainID | None = None
    ) -> dict[str, object] | list[dict[str, object]]:
        """Return native typed annotations for one or every RNA chain."""
        if chain_id is None:
            return [self.get_data(item) for item in self.rna_chain_ids]
        if chain_id not in self.sequences:
            raise ValueError(f"Chain {chain_id} has no sequence information")

        sequence = str(self.sequences_unmod[chain_id])
        chain_info = self.chain_infos[chain_id]
        num_nucleotides = sum(
            len(self.sequences_unmod[chain.subchain_id().split("-")[0]])
            for chain_type in ChainType.ALL_NAS
            for chain in self.chains_of_type[chain_type]
        )
        num_amino_acids = sum(
            len(self.sequences_unmod[chain.subchain_id().split("-")[0]])
            for chain in self.chains_of_type[ChainType.PROTEIN]
        )
        data = {
            "pdb_id": self.pdb_id,
            "asym_id": chain_id,
            "auth_id": self.asym_id_to_auth_id[chain_id],
            "name": self.pdb_name,
            "published": self.published_date,
            "keywords": self.keywords,
            "method": self.method,
            "resolution": self.resolution,
            "organism": chain_info.src_organism,
            "synthetic_organism": chain_info.syn_organism,
            "self_structured": len(self.self_contacts[chain_id])
            > Config3D.MIN_STRUCTURED_CONTACTS,
            "polymer_coverage": self.chain_coverages[ChainType.ANY][chain_id],
            "nucleic_acid_coverage": self.chain_coverages[ChainType.NA][chain_id],
            "num_neighbor_na_chains": len(self.neighbors[chain_id]),
            **{
                name: len(self.chains_of_type[chain_type])
                for name, chain_type in self.__tracked_chain_types.items()
            },
            "num_nucleotides": num_nucleotides,
            "num_amino_acids": num_amino_acids,
            "num_polymer_residues": num_nucleotides + num_amino_acids,
            **{
                name: sorted(self.residues[residue_type])
                for name, residue_type in self.__tracked_residue_types.items()
            },
            "cofactors": sorted(self.cofactors[chain_id]),
            "modified_sequence": str(self.sequences[chain_id]),
            "sequence": sequence,
            "length": len(sequence),
            "fraction_missing": self.get_frac_missing_residues(chain_id),
            "rfam": None,
            "rfam_n_over_l": None,
            "rfam_model_length": None,
            "rfam_model_coverage": None,
            "rfam_bit_score": None,
            "rfam_e_value": None,
        }
        hit = self.fam_hits.get(chain_id)
        best_hit = hit[0] if hit and hit[0] else None
        if best_hit:
            data.update(
                {
                    "rfam": best_hit.name,
                    "rfam_n_over_l": best_hit.n_over_l,
                    "rfam_model_length": best_hit.model_len,
                    "rfam_model_coverage": min(
                        1.0, best_hit.model_len / best_hit.seq_len
                    ),
                    "rfam_bit_score": best_hit.score,
                    "rfam_e_value": best_hit.e_value,
                }
            )
        return data
