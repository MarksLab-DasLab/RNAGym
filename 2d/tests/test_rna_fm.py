import numpy as np

from util.rna_fm import fold
from util.structure import dot_bracket

SEQUENCE = "GGGGAAAACCCC"
EXPECTED = "((((....))))"

contacts = fold(SEQUENCE)
assert contacts.shape == (len(SEQUENCE), len(SEQUENCE))
assert np.isfinite(contacts).all()
structure = dot_bracket(contacts > 0.5)
assert structure == EXPECTED
print(structure)
