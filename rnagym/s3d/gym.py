#!/usr/bin/env python3
"""
`gym.py` is a front end for benchmarking RNA 3D structure prediction tools.
"""

import argparse
from rnagym.s3d.cmd import Analyze, Annotate, EVCouplings, Export, Merge, Split


def main():
    parser = argparse.ArgumentParser(
        description="Front-end for RNAGym 3d benchmarking."
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # --- Add subcommands ---
    # Analyze
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze baseline performance",
    )
    parser_analyze.add_argument("--ignore-cache", action="store_true")

    # Annotate
    subparsers.add_parser(
        "annotate",
        help="Annotate merged PDBs with additional criteria",
    )

    # EVCouplings
    subparsers.add_parser(
        "evcouplings", help="Run EVCouplings on monomer/multimer datasets"
    )

    # Export
    subparsers.add_parser("export", help="Export outputs for sharing")

    # Merge
    subparsers.add_parser("merge", help="Identify unique PDB IDs from data sources")

    # Split
    subparsers.add_parser(
        "split",
        help="Generate the monomer/multimer evaluation sets",
    )

    # --- Run the command ---
    args = parser.parse_args()

    # Map commands to functions
    command_map = {
        "analyze": Analyze.main,
        "annotate": Annotate.main,
        "evcouplings": EVCouplings.main,
        "merge": Merge.main,
        "split": Split.main,
        "export": Export.main,
    }

    if args.command is None:
        # Default pipeline
        print(
            "No command specified. Running default pipeline: merge -> annotate -> split"
        )
        Merge.main()
        Annotate.main()
        Split.main()
    elif args.command == "analyze":
        command_map[args.command](ignore_cache=args.ignore_cache)
    else:
        # Otherwise, run the specified command
        command_map[args.command]()


if __name__ == "__main__":
    main()
