"""Maintain the shared RNAGym sequence registry and similarity clusters."""

import hashlib
import random
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

from rnagym.config import Config2D, ConfigFitness

SEQUENCE_SCHEMA = pl.Schema({"sequence_id": pl.String, "sequence": pl.String})
REGISTRY_SCHEMA = pl.Schema(
    {**SEQUENCE_SCHEMA, "cluster_rep": pl.String, "fold": pl.UInt8}
)
SEARCH_FIELDS = "query,target,fident,alnlen,qcov,tcov,evalue,bits"


def fitness_assays() -> pl.DataFrame:
    """Load ncRNA fitness assay keys and sequences."""
    return (
        pl.read_csv(ConfigFitness.REFERENCE_FILE)
        .filter(pl.col("RNA_TYPE") != "mRNA-coding")
        .select(
            "DMS_ID",
            pl.col("RAW_CONSTRUCT_SEQ")
            .str.to_uppercase()
            .str.replace_all("T", "U")
            .alias("sequence"),
        )
    )


def fitness_sequences() -> pl.DataFrame:
    """Load the unique ncRNA fitness assay sequences."""
    return (
        fitness_assays()
        .select("sequence")
        .unique()
        .with_columns(pl.lit("fitness").alias("modality"))
    )


def load_registry(path: Path, required: bool = True) -> pl.DataFrame:
    """Load and validate the sequence registry."""
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing sequence registry: {path}")
        return pl.DataFrame(schema=REGISTRY_SCHEMA)

    registry = pl.read_parquet(path)
    if registry.schema != REGISTRY_SCHEMA:
        raise RuntimeError(f"Invalid sequence registry schema: {registry.schema}")
    if (
        registry["sequence_id"].n_unique() != registry.height
        or registry["sequence"].n_unique() != registry.height
    ):
        raise RuntimeError("Sequence registry identifiers and sequences must be unique")

    id_numbers = (
        registry["sequence_id"]
        .str.strip_prefix("sequence_")
        .cast(pl.UInt64, strict=False)
    )
    if registry.height and (
        id_numbers.null_count()
        or id_numbers.min() != 0
        or id_numbers.max() != registry.height - 1
    ):
        raise RuntimeError("Sequence registry identifiers must be contiguous")
    return registry


def add_sequences(registry: pl.DataFrame, sequences: pl.Series) -> pl.DataFrame:
    """Append identifiers for sequences not already in the registry."""
    new_sequences = sorted(set(sequences) - set(registry["sequence"]))
    first_id = registry.height
    new_entries = pl.DataFrame(
        {
            "sequence_id": [
                f"sequence_{i:07d}"
                for i in range(first_id, first_id + len(new_sequences))
            ],
            "sequence": new_sequences,
        },
        schema=SEQUENCE_SCHEMA,
    )
    return pl.concat([registry, new_entries])


def write_fasta(sequences: pl.DataFrame, path: Path) -> None:
    """Write sequence IDs and sequences in FASTA format."""
    with path.open("w") as handle:
        handle.writelines(
            f">{sequence_id}\n{sequence}\n"
            for sequence_id, sequence in sequences.iter_rows()
        )


def riboseek(arguments: str) -> None:
    """Run a Riboseek command quietly."""
    subprocess.run(shlex.split(f"riboseek {arguments} -v 1"), check=True)


def read_tsv(path: Path | str, *columns: str) -> pl.DataFrame:
    """Read the leading columns of a headerless Riboseek TSV as strings."""
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=list(columns),
        infer_schema=False,
    ).select(columns)


def clusters_path(work_dir: Path) -> Path:
    """Clusters of one registry version at the configured cuts."""
    identity, coverage = Config2D.MIN_SEQUENCE_IDENTITY, Config2D.MIN_COVERAGE
    return work_dir / f"clusters_id{identity:g}_cov{coverage:g}.parquet"


def prepare(work_dir: Path) -> None:
    """Cluster sequences without library flanks at 80%, then split the
    representatives into blocks for the all-against-all search."""
    db = work_dir / "db"
    if (db / "representatives.index").is_file():
        return
    db.mkdir()
    five, three = Config2D.FLANKS
    trimmed = pl.col("sequence").str.strip_prefix(five).str.strip_suffix(three)
    write_fasta(
        pl.read_parquet(work_dir / "sequences.parquet").with_columns(
            # Keep sequences too short to align without their flanks
            pl.when(trimmed.str.len_chars() >= 20).then(trimmed).otherwise("sequence")
        ),
        work_dir / "trimmed.fasta",
    )
    riboseek(f"createdb {work_dir}/trimmed.fasta {db}/sequences")
    with tempfile.TemporaryDirectory() as temporary_dir:
        riboseek(
            f"linclust {db}/sequences {db}/clusters {temporary_dir} -c 0.8 --min-seq-id 0.8"
        )
    riboseek(f"createsubdb {db}/clusters {db}/sequences {db}/representatives")
    keys = [
        int(line.split("\t")[0])
        for line in (db / "representatives.index").read_text().splitlines()
    ]
    for block in range(Config2D.SEARCH_BLOCKS):
        block_dir = work_dir / "blocks" / str(block)
        block_dir.mkdir(parents=True)
        (block_dir / "keys").write_text(
            "".join(f"{k}\n" for k in keys if k % Config2D.SEARCH_BLOCKS == block)
        )
        riboseek(f"createsubdb {block_dir}/keys {db}/representatives {block_dir}/db")


def search(work_dir: Path, shard: int, num_shards: int) -> None:
    """Search query block i against target block j for this shard's i <= j, so
    each pair is searched once. Each block holds 1 / SEARCH_BLOCKS of the
    representatives, so the E-value cut is divided by SEARCH_BLOCKS."""
    blocks = range(Config2D.SEARCH_BLOCKS)
    pairs = [(i, j) for i in blocks for j in blocks if i <= j]
    (work_dir / "search").mkdir(exist_ok=True)
    for i, j in pairs[shard::num_shards]:
        out = work_dir / "search" / f"{i}_{j}.tsv"
        if out.is_file():
            continue
        query, target = work_dir / f"blocks/{i}/db", work_dir / f"blocks/{j}/db"
        with tempfile.TemporaryDirectory() as temporary_dir:
            riboseek(
                f"search {query} {target} {temporary_dir}/aln {temporary_dir}/tmp "
                f"-a --max-seqs 3000 --num-iterations 1 --prefilter-mode 1 "
                f"--search-type 3 --strand 1 "
                f"-e {Config2D.MAX_EVALUE / Config2D.SEARCH_BLOCKS}"
            )
            riboseek(
                f"convertalis {query} {target} {temporary_dir}/aln {out}.tmp "
                f"--search-type 3 --format-output {SEARCH_FIELDS}"
            )
        Path(f"{out}.tmp").replace(out)


def write_clusters(work_dir: Path) -> None:
    """Greedy set cover of the representatives' hits, as in Riboseek cluster.
    Sequences join their 80% representative's cluster."""
    db = work_dir / "db"
    keys = (
        read_tsv(db / "sequences.lookup", "key", "name")
        .join(read_tsv(db / "representatives.index", "key"), on="key")
        .select(pl.col("key").cast(pl.Int64), "name")
    )
    hits = (
        pl.scan_csv(
            work_dir / "search" / "*.tsv",
            separator="\t",
            has_header=False,
            new_columns=SEARCH_FIELDS.split(","),
            schema_overrides={"query": pl.String, "target": pl.String},
        )
        .filter(
            (pl.col("query") != pl.col("target"))
            & (pl.col("fident") >= Config2D.MIN_SEQUENCE_IDENTITY)
            & (pl.max_horizontal("qcov", "tcov") >= Config2D.MIN_COVERAGE)
        )
        .select(
            "query", "target", pl.col("bits").round().cast(pl.Int64), "fident", "evalue"
        )
        .collect()
    )
    # Mirror hits, since each pair was searched once, and give every representative
    # a self hit so that singletons stay in
    self_hits = keys.select(
        query="name",
        target="name",
        bits=pl.lit(1_000_000, pl.Int64),
        fident=1.0,
        evalue=0.0,
    )
    alignments = (
        pl.concat(
            [self_hits, hits, hits.rename({"query": "target", "target": "query"})],
            how="diagonal",
        )
        .join(keys.rename({"key": "query_key"}), left_on="query", right_on="name")
        .join(keys.rename({"key": "target_key"}), left_on="target", right_on="name")
        .sort("query_key", "target_key", "bits", descending=[False, False, True])
        .unique(["query_key", "target_key"], keep="first", maintain_order=True)
        # Alignment results: target, bits, identity, E-value, then six positions
        .select(
            "query_key",
            "target_key",
            "bits",
            "fident",
            "evalue",
            *(pl.lit(0).alias(f"position_{k}") for k in range(6)),
        )
    )
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        alignments.write_csv(f"{tmp}/aln.tsv", separator="\t", include_header=False)
        riboseek(f"tsv2db {tmp}/aln.tsv {tmp}/aln --output-dbtype 5")
        riboseek(f"clust {db}/representatives {tmp}/aln {tmp}/clusters")
        riboseek(
            f"createtsv {db}/representatives {db}/representatives {tmp}/clusters {tmp}/clusters.tsv"
        )
        riboseek(
            f"createtsv {db}/sequences {db}/sequences {db}/clusters {tmp}/members.tsv"
        )
        clusters = read_tsv(f"{tmp}/clusters.tsv", "cluster_rep", "representative")
        members = read_tsv(f"{tmp}/members.tsv", "representative", "sequence_id")
    members.join(clusters, on="representative").select(
        "sequence_id", "cluster_rep"
    ).write_parquet(clusters_path(work_dir))


def cluster_sequences(sequence_table: pl.DataFrame) -> pl.DataFrame:
    """Load sequence clusters, built per registry version by sh/cluster.sh."""
    sequence_table = sequence_table.select("sequence_id", "sequence").sort(
        "sequence_id"
    )
    fingerprint = hashlib.sha256(
        "".join(f"{i}\t{s}\n" for i, s in sequence_table.iter_rows()).encode()
    ).hexdigest()[:16]
    work_dir = Config2D.CLUSTER_DIR / fingerprint
    if not clusters_path(work_dir).is_file():
        work_dir.mkdir(parents=True, exist_ok=True)
        sequence_table.write_parquet(work_dir / "sequences.parquet")
        raise RuntimeError(
            f"Missing sequence clusters: run `pixi run cluster {work_dir}` in rnagym/s3d"
        )

    assignments = pl.read_parquet(clusters_path(work_dir))
    if not assignments["sequence_id"].sort().equals(sequence_table["sequence_id"]):
        raise RuntimeError("Clusters do not assign every sequence exactly once")
    print(f"Sequence clusters: {assignments['cluster_rep'].n_unique():,}")
    return assignments


def assign_folds(assignments: pl.DataFrame, modalities: pl.DataFrame) -> pl.DataFrame:
    """Balance cluster folds across modality signatures."""
    # Label each cluster by its modalities, for example mapping+pseudobase
    signatures = (
        modalities.join(assignments, on="sequence_id")
        .group_by("cluster_rep")
        .agg(pl.col("modality").unique().sort().str.join("+").alias("signature"))
        .sort(["signature", "cluster_rep"])
    )
    rng = random.Random(Config2D.RANDOM_SEED)
    rows = []
    # Shuffle and distribute each signature evenly across folds
    for group in signatures.partition_by("signature", maintain_order=True):
        cluster_reps = group["cluster_rep"].to_list()
        rng.shuffle(cluster_reps)
        offset = rng.randrange(Config2D.NUM_FOLDS)
        rows.extend(
            (cluster_rep, (i + offset) % Config2D.NUM_FOLDS)
            for i, cluster_rep in enumerate(cluster_reps)
        )
    return pl.DataFrame(
        rows,
        schema={"cluster_rep": pl.String, "fold": pl.UInt8},
        orient="row",
    )


def update_registry(modalities: pl.DataFrame) -> pl.DataFrame:
    """Add modality sequences, then update shared clusters and folds."""
    modalities = modalities.select("sequence", "modality").unique()
    previous = load_registry(Config2D.SEQUENCE_FILE, required=False)
    registry = add_sequences(
        previous.select(SEQUENCE_SCHEMA.names()), modalities["sequence"]
    )
    modalities = modalities.join(registry, on="sequence").select(
        "sequence_id", "modality"
    )

    assignments = cluster_sequences(registry)
    folds = assign_folds(assignments, modalities)
    previous_folds = previous.select(
        "sequence_id", pl.col("fold").alias("previous_fold")
    )
    registry = (
        registry.join(assignments, on="sequence_id")
        .join(folds, on="cluster_rep", how="left")
        .join(previous_folds, on="sequence_id", how="left")
        .with_columns(pl.coalesce("fold", "previous_fold").alias("fold"))
        .drop("previous_fold")
        .sort("sequence_id")
    )
    if registry["fold"].null_count():
        raise RuntimeError("Some sequences did not receive a fold assignment")
    return registry


if __name__ == "__main__":
    # Steps of sh/cluster.sh: STEP WORK_DIR [SHARD NUM_SHARDS]
    step, work_dir, *shard = sys.argv[1:]
    {"prepare": prepare, "search": search, "cluster": write_clusters}[step](
        Path(work_dir), *map(int, shard)
    )
