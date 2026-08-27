"""Test 3D target filtering and cluster-macro scoring."""

import gemmi
import pandas as pd
import polars as pl
import pytest
from rnagym.config import Config3D
from rnagym.s3d.cmd import split as split_cmd
from rnagym.s3d.curation import monomer_filter, quality_filter
from rnagym.s3d.models import homology_columns, prediction_path
from rnagym.s3d.models.utils import prepare_msa
from rnagym.s3d.tasks import score
from rnagym.s3d.util.analysis import _interaction_fidelity, valid_prediction
from rnagym.s3d.util.structure import canonicalize_sequence


def test_3d_benchmark(tmp_path, monkeypatch) -> None:
    """Check quality boundaries and modality-specific aggregation."""
    chain = {
        "fraction_missing": 0.25,
        "method": "X-RAY DIFFRACTION",
        "polymer_coverage": 0.33,
        "resolution": 5.0,
        "self_structured": True,
        "sequence": "ACGU" * 4,
    }

    def matches(row, expression):
        return pl.DataFrame([row]).select(expression).item()

    assert matches(chain, quality_filter())
    assert matches(chain, monomer_filter())
    assert not matches(chain | {"resolution": 5.1}, quality_filter())
    assert not matches(chain | {"sequence": "ACGNNNNN"}, quality_filter())
    assert matches(
        chain | {"method": "SOLUTION NMR", "resolution": None}, quality_filter()
    )
    assert canonicalize_sequence("AUTP?", protein=False) == "AUUNN"
    _, nufold = prediction_path("nu", "sequence_1", False)
    assert nufold.name == "sequence_1_rank_1.pdb"

    annotations = tmp_path / "annotations.parquet"
    pl.DataFrame(
        [
            chain
            | {
                "pdb_id": pdb_id,
                "asym_id": "A",
                "published": published,
                "length": 16,
                "num_polymer_residues": 32,
            }
            for pdb_id, published, chain in (
                ("1aaa", "2024-01-01", chain),
                ("1aab", "2024-01-01", chain | {"polymer_coverage": 0.5}),
                ("1aac", "2020-01-01", chain),
            )
        ]
    ).write_parquet(annotations)
    monkeypatch.setattr(Config3D, "ANNOTATED_CHAINS_FILE", annotations)
    selected = split_cmd.get_split_candidates().select("pdb_id", "type").rows()
    assert selected == [("1aaa", "monomer"), ("1aab", "multimer")]
    assert _interaction_fidelity([], []) == 1
    a, b, c = (("pair", i, i + 1, "cWW") for i in (1, 3, 5))
    assert _interaction_fidelity([], [a]) == 0
    assert _interaction_fidelity([a, b], [a, a, c]) == pytest.approx(2 / 3)

    prediction = tmp_path / "prediction.pdb"
    structure = gemmi.Structure()
    model = gemmi.Model(0)
    chain = gemmi.Chain("A")
    for index in range(2):
        residue = gemmi.Residue()
        residue.name = "A"
        residue.seqid = gemmi.SeqId(index + 1, " ")
        atom = gemmi.Atom()
        atom.name = "C3'"
        atom.pos = gemmi.Position(index, 0, 0)
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.write_minimal_pdb(str(prediction))
    assert valid_prediction(prediction)
    structure[0][0][0][0].pos.x = -999
    structure.write_minimal_pdb(str(prediction))
    assert not valid_prediction(prediction)
    prediction.write_text("not a PDB")
    assert not valid_prediction(prediction)

    identifiers = ["pdb_id", "asym_id", "auth_id"]
    rows = [
        ("monomer", "1aaa", "A", "A", "s1", "c1", 0.4),
        ("monomer", "1aab", "A", "A", "s1", "c1", 0.6),
        ("monomer", "1aac", "A", "A", "s2", "c1", 0.3),
        ("monomer", "1aad", "A", "A", "s3", "c2", 0.5),
        ("multimer", "2aaa", "A", "A", "s4", "c3", 0.4),
        ("multimer", "2aab", "A", "A", "s4", "c3", 0.6),
        ("multimer", "2aac", "A", "A", "s5", "c4", 0.5),
    ]
    references = pd.DataFrame(
        [(*row[:5], row[6]) for row in rows],
        columns=[
            "type",
            *identifiers,
            "sequence_id",
            homology_columns("M")[-1],
        ],
    )
    predictions = pd.DataFrame(
        [
            (*row[0:4], "M", value, value, value)
            for row, value in zip(
                rows,
                [0.2, 0.8, None, 0.6, 0.2, 0.8, 0.6],
            )
            if value is not None
        ],
        columns=["type", *identifiers, "model", "tm_score", "inf_wc", "inf_nwc"],
    )
    target_file = tmp_path / "targets.parquet"
    score_file = tmp_path / "scores.parquet"
    sequence_file = tmp_path / "sequences.parquet"
    references.to_parquet(target_file, index=False)
    predictions.to_parquet(score_file, index=False)
    pd.DataFrame(
        [(row[4], row[5]) for row in rows],
        columns=["sequence_id", "cluster_rep"],
    ).drop_duplicates().to_parquet(sequence_file, index=False)
    monkeypatch.setattr(Config3D, "TARGET_FILE", target_file)
    monkeypatch.setattr(Config3D, "SCORE_FILE", score_file)
    monkeypatch.setattr(Config3D, "SEQUENCE_FILE", sequence_file)
    monkeypatch.setattr(score, "MODELS", {"M": "Model"})

    monomer = score.score_dataset("monomer").iloc[0]
    multimer = score.score_dataset("multimer").iloc[0]
    assert monomer.tm_score == pytest.approx(0.5)
    assert monomer.completed == 3
    assert multimer.tm_score == pytest.approx(0.55)
    assert multimer.completed == 3


def test_prepare_msa_preserves_headers(tmp_path, monkeypatch) -> None:
    """Normalize RNA sequences without changing MSA identifiers."""
    source = tmp_path / "sequence_1.a3m"
    source.write_text(">8T3D_X\nATX\n>hit_T\naTx\n")
    monkeypatch.setattr(Config3D, "MSA_DIR", tmp_path)

    output, sequence = prepare_msa("sequence_1", tmp_path / "output.a3m")

    assert output.read_text() == ">8T3D_X\nAUA\n>hit_T\naUa\n"
    assert sequence == "AUA"
