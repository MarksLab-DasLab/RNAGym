"""Generate missing target-to-training US-align comparisons."""

import sys

import polars as pl

from rnagym.config import Config3D
from rnagym.s3d.models import BASELINES, homology_columns
from rnagym.s3d.util.analysis import add_tm_id, prep_usalign

IDENTIFIERS = ["pdb_id", "asym_id"]
HOMOLOGY_COLUMNS = [column for model in BASELINES for column in homology_columns(model)]
ANNOTATION_SCHEMA = pl.Schema(
    {
        column: pl.Float64 if column.endswith("_score") else pl.String
        for column in [*IDENTIFIERS, *HOMOLOGY_COLUMNS]
    }
)


def main() -> None:
    """Process one shard of 3D targets."""
    if sys.argv[1:] == ["prepare"]:
        prep_usalign()
        return

    shard, num_shards = map(int, sys.argv[1:])
    targets = (
        pl.read_parquet(Config3D.TARGET_FILE)
        .sort("length", "pdb_id", "asym_id", descending=[True, False, False])
        .gather_every(num_shards, offset=shard)
    )
    Config3D.USALIGN_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    for target in targets.iter_rows(named=True):
        name = f"{target['pdb_id']}_{target['asym_id']}"
        output = Config3D.USALIGN_ANNOTATION_DIR / f"{name}.parquet"
        source = Config3D.chain_file(target["pdb_id"], target["asym_id"])
        dependencies = (Config3D.USALIGN_REFERENCES_FILE, source)
        fresh = (
            output.is_file()
            and output.stat().st_size
            and output.stat().st_mtime
            >= max(path.stat().st_mtime for path in dependencies)
        )
        if fresh and pl.read_parquet_schema(output) == ANNOTATION_SCHEMA:
            continue
        result = add_tm_id(pl.from_dicts([target]))
        temporary = output.with_suffix(".parquet.tmp")
        result.select(*IDENTIFIERS, *HOMOLOGY_COLUMNS).cast(
            ANNOTATION_SCHEMA
        ).write_parquet(temporary)
        temporary.replace(output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
