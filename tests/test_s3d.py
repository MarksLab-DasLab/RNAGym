"""Reproduce the 3D benchmark workflow on a small real-data subset."""

from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from rnagym.config import Config2D, Config3D
from rnagym.s3d.cmd import split
from rnagym.s3d.tasks import score
from rnagym.sequences import cluster_dir, clusters_path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "s3d"


def test_3d_workflow(tmp_path, monkeypatch) -> None:
    """Reproduce target selection, cluster-macro scoring, and leaderboard output."""
    monkeypatch.setattr(Config2D, "MAPPING_FILE", tmp_path / Config2D.MAPPING_FILE.name)
    monkeypatch.setattr(
        Config2D, "STRUCTURE_FILE", tmp_path / Config2D.STRUCTURE_FILE.name
    )
    monkeypatch.setattr(
        Config2D, "SEQUENCE_FILE", tmp_path / Config2D.SEQUENCE_FILE.name
    )
    monkeypatch.setattr(
        Config3D, "ANNOTATED_CHAINS_FILE", FIXTURE_DIR / "annotated_chains.parquet"
    )
    monkeypatch.setattr(Config3D, "LEADERBOARD_DIR", tmp_path / "leaderboard")
    monkeypatch.setattr(
        Config3D, "LEADERBOARD_FILE", Config3D.LEADERBOARD_DIR / "leaderboard.csv"
    )
    monkeypatch.setattr(
        Config3D, "LEADERBOARD_README", Config3D.LEADERBOARD_DIR / "README.md"
    )
    monkeypatch.setattr(Config3D, "SCORE_FILE", FIXTURE_DIR / "scores.parquet")
    monkeypatch.setattr(Config3D, "SEQUENCE_FILE", Config2D.SEQUENCE_FILE)
    monkeypatch.setattr(Config3D, "TARGET_FILE", tmp_path / "rnagym_3d.parquet")
    monkeypatch.setattr(Config3D, "USALIGN_ANNOTATION_DIR", FIXTURE_DIR)

    pl.DataFrame(schema={"sequence": pl.String}).write_parquet(Config2D.MAPPING_FILE)
    pl.DataFrame(schema={"uid": pl.String, "sequence": pl.String}).write_parquet(
        Config2D.STRUCTURE_FILE
    )
    registry = pl.read_parquet(FIXTURE_DIR / "sequences.parquet")
    registry.write_parquet(Config2D.SEQUENCE_FILE)
    # Reuse the fixture's clusters instead of the all-against-all search
    monkeypatch.setattr(Config2D, "CLUSTER_DIR", tmp_path / "clusters")
    clusters = clusters_path(cluster_dir(registry))
    clusters.parent.mkdir(parents=True)
    registry.select("sequence_id", "cluster_rep").write_parquet(clusters)
    Config3D.LEADERBOARD_DIR.mkdir()
    Config3D.LEADERBOARD_README.write_text(
        "# Test leaderboard\n\n"
        "<!-- BEGIN GENERATED TABLES -->\n"
        "<!-- END GENERATED TABLES -->\n"
    )
    monkeypatch.setattr(
        split,
        "fitness_sequences",
        lambda: pl.DataFrame(schema={"sequence": pl.String, "modality": pl.String}),
    )

    split.main()
    score.main()

    assert_frame_equal(
        pl.read_parquet(Config3D.TARGET_FILE)
        .select("pdb_id", "asym_id", "type")
        .sort("pdb_id", "asym_id"),
        pl.read_csv(FIXTURE_DIR / "targets.csv"),
    )
    assert_frame_equal(
        pl.read_csv(Config3D.LEADERBOARD_FILE),
        pl.read_csv(FIXTURE_DIR / "leaderboard.csv"),
    )
    assert (
        Config3D.LEADERBOARD_README.read_text()
        == (FIXTURE_DIR / "leaderboard.md").read_text()
    )
