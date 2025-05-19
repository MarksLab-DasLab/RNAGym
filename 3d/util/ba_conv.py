###############################################################################
# `ba_conv.py`: Conversion for chains in biological assembly != 1
###############################################################################
# Converts a chain identifier from biological assembly 2 to a homologous
# identifier in assembly 1.  This is necessary because annotate only considers
# the first assembly.
BA2_TO_BA1 = {
    "8FON": {"XX": "QX"},
    "8FOM": {"XX": "QX"},
    "8F5G": {"F": "D"},
}

BA1_TO_BA2 = {
    "8FON": {"QX": "XX"},
    "8FOM": {"QX": "XX"},
    "8F5G": {"D": "F"},
}
