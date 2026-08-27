"""Generate missing target-to-training US-align comparisons."""

import sys

import pandas as pd

from rnagym.config import Config3D
from rnagym.s3d.models import BASELINES, homology_columns
from rnagym.s3d.util.analysis import add_tm_id, prep_usalign

IDENTIFIERS = ["pdb_id", "asym_id"]
HOMOLOGY_COLUMNS = [column for model in BASELINES for column in homology_columns(model)]


def main() -> None:
    """Process one shard of 3D targets."""
    if sys.argv[1:] == ["prepare"]:
        prep_usalign()
        return

    shard, num_shards = map(int, sys.argv[1:])
    targets = (
        pd.read_parquet(Config3D.TARGET_FILE)
        .sort_values("length", ascending=False)
        .iloc[shard::num_shards]
    )
    Config3D.USALIGN_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    for _, target in targets.iterrows():
        name = f"{target.pdb_id}_{target.asym_id}"
        output = Config3D.USALIGN_ANNOTATION_DIR / f"{name}.parquet"
        source = Config3D.chain_file(target.pdb_id, target.asym_id)
        dependencies = (Config3D.USALIGN_REFERENCES_FILE, source)
        fresh = (
            output.is_file()
            and output.stat().st_size
            and output.stat().st_mtime
            >= max(path.stat().st_mtime for path in dependencies)
        )
        if fresh and set([*IDENTIFIERS, *HOMOLOGY_COLUMNS]).issubset(
            pd.read_parquet(output).columns
        ):
            continue
        result = target.to_frame().T
        add_tm_id(result)
        temporary = output.with_suffix(".parquet.tmp")
        result[[*IDENTIFIERS, *HOMOLOGY_COLUMNS]].to_parquet(temporary, index=False)
        temporary.replace(output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
