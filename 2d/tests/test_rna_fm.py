import numpy as np

from models.rna_fm import fold
from models.utils import dot_bracket

SEQUENCE = "GGGGAAAACCCC"
EXPECTED = "((((....))))"

contacts = fold(SEQUENCE)
assert contacts.shape == (len(SEQUENCE), len(SEQUENCE))
assert np.isfinite(contacts).all()
structure = dot_bracket(contacts > 0.5)
assert structure == EXPECTED
print(structure)
