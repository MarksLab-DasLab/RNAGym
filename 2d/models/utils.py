import numpy as np


def dot_bracket(contact: np.ndarray) -> str:
    structure = ["."] * len(contact)
    for i, j in zip(*np.where(np.triu(contact, k=1))):
        structure[i] = "("
        structure[j] = ")"
    return "".join(structure)
