import numpy as np

from util.structure import dot_bracket
from util.ufold import fold

SEQUENCE = "GGGGAAAACCCC"
EXPECTED = "((((....))))"

contacts = fold(SEQUENCE)
assert contacts.shape == (len(SEQUENCE), len(SEQUENCE))
assert np.isfinite(contacts).all()
structure = dot_bracket(contacts > 0.5)
assert structure == EXPECTED
print(structure)
