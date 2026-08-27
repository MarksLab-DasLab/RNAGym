#!/usr/bin/env python3

###############################################################################
# `types.py`:  Useful types for RNA Gym 3D
###############################################################################
from typing import Dict, Set, Tuple

import gemmi

# ---  Structure ---
AccessionID = str
ChainID = str
EntityID = int
MonomerID = str
PdbID = str
ResName = str
Sequence = str

# --- Analysis ---
AtomName = str
AltID = str
Category = str
ContactMap = Dict[gemmi.Residue, Set[gemmi.Residue]]
Fr3dResID = str
InsertionCode = str
ModelID = int
ResName = str
ResID = int
SymmetryOp = str

# int := # interactions crossed by the contact, i.e. the degree of
# non-nestedness.  0 for nested base pairs
Fr3dContactID = Tuple[Fr3dResID, Fr3dResID, int]
