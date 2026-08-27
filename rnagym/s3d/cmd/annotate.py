"""Annotate candidate PDB RNA chains for filtering."""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import polars as pl
from tqdm import tqdm

from rnagym.config import Config3D
from rnagym.s3d.util.sequence import rfam_families, rna3db_hits
from rnagym.s3d.util.structure import StructureInfo


def process_pdb(pdb_id: str) -> list[dict[str, object]]:
    """Annotate every RNA chain in one PDB entry."""
    structure = StructureInfo.from_pdb_id(pdb_id)
    return [] if structure is None else structure.get_data()


def main() -> None:
    """Write native typed annotations for every RNA3DB chain."""
    chains = json.loads(Config3D.RNA3DB_PARSE_FILE.read_text())
    pdb_ids = sorted({chain.partition("_")[0] for chain in chains})
    # Load immutable lookups before forking so workers share the same pages
    rfam_families()
    rna3db_hits()
    workers = len(os.sched_getaffinity(0))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = [
            row
            for annotations in tqdm(
                executor.map(process_pdb, pdb_ids, chunksize=1),
                total=len(pdb_ids),
                desc="Annotating RNA3DB",
            )
            for row in annotations
        ]

    data = pl.from_dicts(rows, infer_schema_length=None)
    list_columns = [
        name for name, dtype in data.schema.items() if isinstance(dtype, pl.List)
    ]
    data = data.with_columns(pl.col(list_columns).cast(pl.List(pl.String))).sort(
        "pdb_id", "auth_id"
    )
    output = Config3D.ANNOTATED_CHAINS_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    data.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(output)
    print(f"Wrote {data.height:,} chains to {output}")


if __name__ == "__main__":
    main()
