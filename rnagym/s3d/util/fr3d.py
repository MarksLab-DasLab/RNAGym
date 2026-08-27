###############################################################################
# `fr3d.py`: Data required/useful for interfacing with FR3D Python
###############################################################################

from itertools import product

# W := Watson-Crick
# H := Hoogsteen
# S := sugar
# R := ribose (unclear how this differs from sugar)
VALID_BASEPAIRS = ("W", "H", "S", "R")


def get_basepair_labels(bp1: str, bp2: str) -> [str]:
    """
    Returns a list of strings of all valid FR3D permutations of two basepair
    categories.
    """
    if not (bp1 in VALID_BASEPAIRS and bp2 in VALID_BASEPAIRS):
        raise ValueError("bp1 and bp2 must be in VALID_BASEPAIRS")

    # cis- basepairs
    basepair_labels = [f"c{bp1}{bp2}", f"c{bp2}{bp1}"]

    # trans- basepairs
    for bp_label in basepair_labels[:]:
        basepair_labels.append("t" + bp_label[1:])

    # Leontis-Westhoff notation where uppercase has "priority"
    for bp_label in basepair_labels[:]:
        basepair_labels.append(bp_label[0] + bp_label[1].lower() + bp_label[2])
        basepair_labels.append(bp_label[0] + bp_label[1] + bp_label[2].lower())

    # Anionic basepairs
    for bp_label in basepair_labels[:]:
        basepair_labels.append(bp_label + "a")

    return basepair_labels


BASEPAIR_LABELS = set()
for bp1, bp2 in product(VALID_BASEPAIRS, repeat=2):
    BASEPAIR_LABELS.update(get_basepair_labels(bp1, bp2))

BASEPAIR_LABELS = BASEPAIR_LABELS | {"cBW", "cWB"}  # Bifurcated pairs

# 3 := sugar 3' face
# 5 := sugar 5' face
# O#' := sugar oxygen O#'
STACKING_LABELS = {
    "s33",
    "s35",
    "s53",
    "s55",
    "sO1'3",
    "s3O1'",
    "sO1'5",
    "s5O1'",
    "sO2'3",
    "s3O2'",
    "sO2'5",
    "s5O2'",
    "s3O3'",
    "sO3'3",
    "s5O3'",
    "sO3'5",
    "sO4'3",
    "s3O4'",
    "sO4'5",
    "s5O4'",
    "s3O5'",
    "sO5'3",
    "s5O5'",
    "sO5'5",
    "s3OP1",
    "sOP13",
    "s5OP1",
    "sOP15",
    "s3OP2",
    "sOP23",
    "s5OP2",
    "sOP25",
}


# See: https://rna.bgsu.edu/FR3D/BasePhosphates/
# NOTE(MCA): I have not found a matching resource for Base-Ribose interactions
#   but assume that, much like the Base-Phosphate interactions, they are just
#   defined by similarity scores to some arbitrary Base-Ribose motifs
BASE_RIBOSE_LABELS = {f"{i}BR" for i in range(10)}
PHOSPHATE_LABELS = {f"{i}BPh" for i in range(10)}

# Unable to identify what these represent
UNKNOWN_LABELS = {
    "p_1",
    "p_2",
    "p_4",
    "p_9",
    "cp",
}

# NOTE(MCA): The prefix "n" means "near". This means that the interaction did
#   not meet the cutoffs for any interaction, but was closest to whatever
#   interaction is described by the suffix
CONTACT_LABELS = (
    BASEPAIR_LABELS
    | BASE_RIBOSE_LABELS
    | STACKING_LABELS
    | PHOSPHATE_LABELS
    | UNKNOWN_LABELS
)
CONTACT_LABELS.update([f"n{label}" for label in CONTACT_LABELS])

# Define "2D" labels as any traditional, Watson-Crick basepairs.  "3D"
# interactions include all other types of interactions
CONTACT_LABELS_2D = set()
CONTACT_LABELS_2D.update(get_basepair_labels("W", "W"))
CONTACT_LABELS_3D = CONTACT_LABELS - CONTACT_LABELS_2D

# Dictionary used by `annotate_nt_nt_in_structure`.  The keys are a
# comprehensive list of all possible interaction types.  The empty list values
# indicate that all NT-NT interactions of each category should be considered
NT_NT_CATEGORIES = {
    "backbone": [],
    "basepair": [],
    "basepair_detail": [],  # includes bifurcated pairs
    "coplanar": [],
    "covalent": [],
    "near": [],
    "sO": [],
    "sugar_ribose": [],
    "stacking": [],
}
