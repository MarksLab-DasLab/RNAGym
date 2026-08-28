"""Canonicalize PDB polymer sequences."""

import gemmi

from rnagym.s3d.util import Residues


def canonical_residue(residue: str, polymer_type: gemmi.PolymerType) -> str:
    """Convert one PDB chemical component into one canonical residue.

    Parameters
    ----------
    residue : str
        PDB chemical component name.
    polymer_type : gemmi.PolymerType
        Polymer type that determines the alphabet and unknown token.

    Returns
    -------
    str
        One canonical residue.
    """
    protein = polymer_type in {gemmi.PolymerType.PeptideD, gemmi.PolymerType.PeptideL}
    unknown = "X" if protein else "N"
    # Microheterogeneous alternatives still occupy one polymer position
    if "," in residue:
        return unknown

    code = (
        residue
        if len(residue) == 1
        else (
            Residues.ModNA.get(residue)
            or Residues.ModProtein.get(residue)
            or Residues.ProteinTo1Letter.get(residue)
            or Residues.DNATo1Letter.get(residue)
            or gemmi.find_tabulated_residue(residue).one_letter_code
        )
    )
    code = code.upper()
    # ModNA denotes modified DNA bases as DA, DC, DG, DT, or DU
    if len(code) == 2 and code.startswith("D"):
        code = code[1]
    if protein:
        return code if len(code) == 1 and code.isalpha() else unknown
    if polymer_type in {gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid}:
        code = code.replace("T", "U")
    elif polymer_type == gemmi.PolymerType.Dna:
        code = code.replace("U", "T")
    return code if code in "ACGTU" else unknown


def canonical_polymer_sequence(entity: gemmi.Entity) -> str:
    """Return exactly one canonical residue per polymer position."""
    return "".join(
        canonical_residue(residue, entity.polymer_type)
        for residue in entity.full_sequence
    )
