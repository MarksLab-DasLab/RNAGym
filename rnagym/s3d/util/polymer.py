"""Canonicalize PDB polymer sequences."""

import csv
from functools import cache
from pathlib import Path

import gemmi

from rnagym.config import Config3D
from rnagym.s3d.util import Residues

BASES = {"A", "C", "G", "T", "U"}
# Every CCD nucleotide outside this set counts as modified
STANDARD_NUCLEOTIDES = {
    "A",
    "C",
    "G",
    "I",
    "N",
    "U",
    "DA",
    "DC",
    "DG",
    "DI",
    "DN",
    "DT",
    "DU",
}


def write_ccd_table(archive: Path, output: Path) -> None:
    """Extract each component's one-letter code, parent, and type from the CCD.

    Parsing the full dictionary takes seconds and gigabytes, so lookups read
    this small table instead.
    """
    rows = []
    for block in gemmi.cif.read(str(archive)):
        fields = (
            block.find_value("_chem_comp.one_letter_code"),
            block.find_value("_chem_comp.mon_nstd_parent_comp_id"),
            block.find_value("_chem_comp.type"),
        )
        rows.append([block.name, *(gemmi.cif.as_string(field) for field in fields)])
    # sh/fetch-ccd.sh holds a lock, so one temporary name is safe
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)
    temporary.chmod(0o664)
    temporary.replace(output)


@cache
def _ccd() -> dict[str, tuple[str, str, str]]:
    """Load each component's one-letter code, parent, and type."""
    if not Config3D.CCD_TABLE_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {Config3D.CCD_TABLE_FILE}, run `pixi run ccd` first"
        )
    with Config3D.CCD_TABLE_FILE.open(newline="") as handle:
        return {
            component: (code, parent, kind)
            for component, code, parent, kind in csv.reader(handle, delimiter="\t")
        }


def ccd_base(component: str) -> str:
    """Return the base the CCD assigns one nucleotide component.

    The CCD is the only authority, so there are no hand-written exceptions.
    A component's own one-letter code decides, and when that is not a base,
    its parent component's code does. Anything else is N. For example:

    - GTP is G, from its one-letter code.
    - Inosine (I) stays N: the CCD gives it the code I and no parent.
    - A1LZ3 is U as the CCD says, although its base is a cytidine (ava2C).
    """
    seen = set()
    # Parents can be multi-component lists such as "DT, DT", which are not keys
    while component in _ccd() and component not in seen:
        seen.add(component)
        code, parent, _ = _ccd()[component]
        if code.upper() in BASES:
            return code.upper()
        component = parent
    return "N"


def nucleic_acid(component: str) -> str:
    """Return the nucleic acid, RNA or DNA, a CCD component links into.

    Any other component, including one absent from the CCD, returns "".
    L-RNA and terminal types such as "RNA OH 5 PRIME TERMINUS" count.
    """
    kind = _ccd().get(component, ("", "", ""))[2].upper()
    return next((acid for acid in ("RNA", "DNA") if acid in kind), "")


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

    # Nucleotides take their base from the CCD alone, as ccd_base describes
    if not protein:
        code = ccd_base(residue)
        if polymer_type in {gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid}:
            code = code.replace("T", "U")
        elif polymer_type == gemmi.PolymerType.Dna:
            code = code.replace("U", "T")
        return code if code in BASES else unknown

    code = (
        residue
        if len(residue) == 1
        else (
            Residues.ModProtein.get(residue)
            or Residues.ProteinTo1Letter.get(residue)
            or gemmi.find_tabulated_residue(residue).one_letter_code
        )
    )
    code = code.upper()
    return code if len(code) == 1 and code.isalpha() else unknown


def canonical_polymer_sequence(entity: gemmi.Entity) -> str:
    """Return exactly one canonical residue per polymer position."""
    return "".join(
        canonical_residue(residue, entity.polymer_type)
        for residue in entity.full_sequence
    )
