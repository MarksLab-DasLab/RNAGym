#!/usr/bin/env python3

###############################################################################
# `analyze.py`: Analyze RNAGym baseline outputs.
###############################################################################
from util.analysis import get_baseline_scores


def main(ignore_cache=False):
    get_baseline_scores(print_status=True, ignore_cache=ignore_cache)


if __name__ == "__main__":
    main()
