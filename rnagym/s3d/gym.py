"""Run the 3D dataset curation pipeline."""

import argparse

from rnagym.s3d.cmd import Annotate, Split


def main() -> None:
    """Run all curation steps or one requested step."""
    parser = argparse.ArgumentParser(description="Curate the RNAGym 3D benchmark")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.add_parser("annotate", help="Annotate RNA3DB chains")
    subparsers.add_parser("split", help="Select monomer and multimer targets")
    args = parser.parse_args()
    steps = {"annotate": Annotate.main, "split": Split.main}
    selected = steps.values() if args.command is None else (steps[args.command],)
    for step in selected:
        step()


if __name__ == "__main__":
    main()
