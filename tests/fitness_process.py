"""Run a nested inference process for the reproduction timeout regression."""

import shlex
import sys
from pathlib import Path

from rnagym.fitness.tasks import check_published


def run_one(environment, deadline):
    """Record the inference PID and wait for cancellation."""
    yield {"model": "started"}
    script = f"echo $$ > {shlex.quote(sys.argv[1])}\nexec sleep 30"
    check_published.run_command(f"bash -c {shlex.quote(script)}", Path.cwd(), deadline)


if __name__ == "__main__":
    check_published.run_one = run_one
    check_published.main(["rna-fm", "--report", sys.argv[2]])
