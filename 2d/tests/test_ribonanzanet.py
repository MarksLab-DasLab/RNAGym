import subprocess
import sys

RIBONANZANET_SCRIPT = ".pixi/model-sources/rnet-inference/src/rnet_2d.py"
SEQUENCE = "GGGGAAAACCCC"
EXPECTED = "((((....))))"

result = subprocess.run(
    [sys.executable, RIBONANZANET_SCRIPT, SEQUENCE],
    check=True,
    capture_output=True,
    text=True,
)
structure = next(
    line.removeprefix("structure:")
    for line in result.stdout.splitlines()
    if line.startswith("structure:")
)
assert structure == EXPECTED
print(structure)
