"""Generate missing target-to-training US-align comparisons."""

import sys

import pandas as pd

from rnagym.config import Config3D
from rnagym.s3d.util.analysis import add_tm_id


def main() -> None:
    """Process one shard of 3D targets."""
    shard, num_shards = map(int, sys.argv[1:])
    targets = pd.read_parquet(Config3D.TARGET_FILE).iloc[shard::num_shards]
    Config3D.USALIGN_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    for _, target in targets.iterrows():
        name = f"{target['PDB ID'].lower()}_{target['Asym. Chain ID']}"
        output = Config3D.USALIGN_ANNOTATION_DIR / f"{name}.parquet"
        if output.is_file() and output.stat().st_size:
            continue
        result = target.to_frame().T
        add_tm_id(result)
        columns = [
            "PDB ID",
            "Asym. Chain ID",
            *[column for column in result if "TM Homolog" in column],
        ]
        temporary = output.with_suffix(".parquet.tmp")
        result[columns].to_parquet(temporary, index=False)
        temporary.replace(output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
