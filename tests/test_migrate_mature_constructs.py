"""Rewrite assay tables onto mature constructs without losing a measurement."""

import csv
from pathlib import Path

import pytest

from rnagym.fitness.tasks.migrate_mature_constructs import (
    RECIPES,
    build_mapping,
    main,
    mature_sequence,
    parse_mutations,
)

REFERENCE_COLUMNS = ("DMS_ID", "RNA_TYPE", "RAW_CONSTRUCT_SEQ", "Notes")
ASSAY_COLUMNS = ("mutant", "DMS_score", "sequence")
COMPLEMENT = {"A": "G", "C": "U", "G": "A", "U": "C"}


def construct(length):
    """Build a deterministic RNA construct of one length."""
    return ("ACGUUGCA" * (length // 8 + 1))[:length]


def substitute(sequence, position, base):
    """Apply one one-based substitution to a sequence."""
    return sequence[: position - 1] + base + sequence[position:]


def write_csv(path, columns, rows, newline, bom):
    """Write one CSV in the byte conventions under test."""
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), lineterminator=newline
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    """Read one CSV back as a list of dictionaries."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def benchmark(tmp_path):
    """Write a miniature reference sheet and the assay tables it describes."""
    assays = tmp_path / "assays"
    assays.mkdir()
    constructs = {
        "Guy_2014_tRNA": construct(105),
        "Domingo_2018_tRNA": construct(72),
        "Li_2016_tRNA": construct(72),
        "Kobori_2018_ribozyme": construct(40).replace("U", "T"),
    }
    reference = [
        {
            "DMS_ID": assay,
            "RNA_TYPE": "tRNA",
            "RAW_CONSTRUCT_SEQ": sequence,
            "Notes": "Existing note." if assay == "Guy_2014_tRNA" else "",
        }
        for assay, sequence in constructs.items()
    ]
    write_csv(
        tmp_path / "reference.csv", REFERENCE_COLUMNS, reference, "\r\n", bom=True
    )

    for assay in RECIPES:
        sequence = constructs[assay]
        positions = [1, 39, 54, 72] if assay == "Guy_2014_tRNA" else [1, 72]
        rows = [
            {
                "mutant": f"{sequence[p - 1]}{p}{COMPLEMENT[sequence[p - 1]]}",
                "DMS_score": f"0.{p:03d}",
                "sequence": substitute(sequence, p, COMPLEMENT[sequence[p - 1]]),
            }
            for p in positions
        ]
        write_csv(assays / f"{assay}.csv", ASSAY_COLUMNS, rows, "\n", bom=False)
    return tmp_path, assays, constructs


def run(tmp_path, assays, extra=()):
    """Run the migration into a fresh output directory."""
    out = tmp_path / "out"
    return (
        main(
            [
                "--assays-dir",
                str(assays),
                "--reference",
                str(tmp_path / "reference.csv"),
                "--out-dir",
                str(out),
                *extra,
            ]
        ),
        out,
    )


def test_guy_mapping_keeps_every_scored_position():
    """Guy's two exons map onto 75 contiguous mature positions."""
    mapping = build_mapping(RECIPES["Guy_2014_tRNA"].segments, 105)
    assert len(mapping) == 75
    assert mapping[1] == 1
    assert mapping[39] == 39
    # The 14 nt intron is spliced out, so exon 2 starts immediately after exon 1
    assert mapping[54] == 40
    assert mapping[89] == 75
    assert all(position not in mapping for position in range(40, 54))
    assert all(position not in mapping for position in range(90, 106))


def test_mature_sequence_adds_the_encoded_end():
    """A mature tRNA is the retained segments plus the CCA no gene encodes."""
    recipe = RECIPES["Domingo_2018_tRNA"]
    assert mature_sequence(construct(72), recipe.segments, recipe.append) == (
        construct(72) + "CCA"
    )


def test_parse_mutations_reads_dna_labels_as_rna():
    """A label written in DNA names the same RNA substitution."""
    assert parse_mutations("T5C") == [("U", 5, "C")]
    assert parse_mutations("") == []


def test_parse_mutations_rejects_a_contradictory_label():
    """Two tokens for one position would let the last one win silently."""
    with pytest.raises(ValueError, match="more than once"):
        parse_mutations("A5C,A5G")


def test_migration_rewrites_constructs_labels_and_sequences(benchmark):
    """Every corrected assay is rewritten onto its mature construct."""
    tmp_path, assays, constructs = benchmark
    status, out = run(
        tmp_path, assays, ("--out-reference", str(tmp_path / "corrected.csv"))
    )
    assert status == 0

    guy = read_csv(out / "Guy_2014_tRNA.csv")
    mature = mature_sequence(
        constructs["Guy_2014_tRNA"], RECIPES["Guy_2014_tRNA"].segments, "CCA"
    )
    # Position 54 is the first base of exon 2 and becomes position 40
    assert [row["mutant"][1:-1] for row in guy] == ["1", "39", "40", "58"]
    assert [row["DMS_score"] for row in guy] == ["0.001", "0.039", "0.054", "0.072"]
    for row in guy:
        position = int(row["mutant"][1:-1])
        assert row["sequence"] == substitute(mature, position, row["mutant"][-1])

    corrected = {row["DMS_ID"]: row for row in read_csv(tmp_path / "corrected.csv")}
    assert len(corrected["Guy_2014_tRNA"]["RAW_CONSTRUCT_SEQ"]) == 78
    assert len(corrected["Domingo_2018_tRNA"]["RAW_CONSTRUCT_SEQ"]) == 75
    assert corrected["Guy_2014_tRNA"]["Notes"].startswith("Existing note. Mature tRNA")
    assert corrected["Domingo_2018_tRNA"]["Notes"] == "Mature tRNA: CCA added."


def test_transcription_changes_the_reference_sheet_only(benchmark):
    """A DNA-stored construct becomes RNA without touching its assay table."""
    tmp_path, assays, constructs = benchmark
    status, out = run(
        tmp_path, assays, ("--out-reference", str(tmp_path / "corrected.csv"))
    )
    assert status == 0
    corrected = {row["DMS_ID"]: row for row in read_csv(tmp_path / "corrected.csv")}
    assert "T" not in corrected["Kobori_2018_ribozyme"]["RAW_CONSTRUCT_SEQ"]
    assert corrected["Kobori_2018_ribozyme"]["RAW_CONSTRUCT_SEQ"] == constructs[
        "Kobori_2018_ribozyme"
    ].replace("T", "U")
    assert not (out / "Kobori_2018_ribozyme.csv").exists()


def test_each_table_keeps_the_byte_conventions_it_was_read_with(benchmark):
    """The assay tables stay LF without a BOM and the sheet stays CRLF with one."""
    tmp_path, assays, _ = benchmark
    status, out = run(
        tmp_path, assays, ("--out-reference", str(tmp_path / "corrected.csv"))
    )
    assert status == 0
    assay_bytes = (out / "Guy_2014_tRNA.csv").read_bytes()
    assert not assay_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in assay_bytes
    sheet_bytes = (tmp_path / "corrected.csv").read_bytes()
    assert sheet_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in sheet_bytes


def test_a_wild_type_row_survives_the_migration(benchmark):
    """An empty label is a wild-type control, not a malformed row."""
    tmp_path, assays, constructs = benchmark
    sequence = constructs["Domingo_2018_tRNA"]
    write_csv(
        assays / "Domingo_2018_tRNA.csv",
        ASSAY_COLUMNS,
        [{"mutant": "", "DMS_score": "1.0", "sequence": sequence}],
        "\n",
        bom=False,
    )
    status, out = run(tmp_path, assays)
    assert status == 0
    rows = read_csv(out / "Domingo_2018_tRNA.csv")
    assert rows == [{"mutant": "", "DMS_score": "1.0", "sequence": sequence + "CCA"}]


@pytest.mark.parametrize(
    "label",
    [
        pytest.param("A40C", id="inside the spliced intron"),
        pytest.param("A200C", id="beyond the construct"),
    ],
)
def test_nothing_is_written_when_one_assay_fails(benchmark, label, capsys):
    """A partial batch would leave new, stale and missing tables side by side."""
    tmp_path, assays, constructs = benchmark
    sequence = constructs["Guy_2014_tRNA"]
    write_csv(
        assays / "Guy_2014_tRNA.csv",
        ASSAY_COLUMNS,
        [{"mutant": label, "DMS_score": "1.0", "sequence": ""}],
        "\n",
        bom=False,
    )
    status, out = run(tmp_path, assays)
    assert status == 1
    assert "nothing was written" in capsys.readouterr().err
    # Domingo and Li are valid, and must not have been written either
    assert not out.exists()
    assert len(read_csv(assays / "Domingo_2018_tRNA.csv")) == 2
    assert len(sequence) == 105


def test_a_label_disagreeing_with_the_construct_fails(benchmark, capsys):
    """A wild-type base that is not in the construct means the inputs disagree."""
    tmp_path, assays, constructs = benchmark
    sequence = constructs["Li_2016_tRNA"]
    wrong = COMPLEMENT[sequence[0]]
    write_csv(
        assays / "Li_2016_tRNA.csv",
        ASSAY_COLUMNS,
        [{"mutant": f"{wrong}1A", "DMS_score": "1.0", "sequence": ""}],
        "\n",
        bom=False,
    )
    status, _ = run(tmp_path, assays)
    assert status == 1
    assert "disagrees with construct base" in capsys.readouterr().err


def test_writing_into_the_source_directory_is_refused(benchmark, capsys):
    """Writing in place would overwrite the originals the migration reads."""
    tmp_path, assays, _ = benchmark
    status = main(
        [
            "--assays-dir",
            str(assays),
            "--reference",
            str(tmp_path / "reference.csv"),
            "--out-dir",
            str(assays),
        ]
    )
    assert status == 1
    assert "refusing to write into the source directory" in capsys.readouterr().err


def test_a_duplicate_assay_in_the_reference_sheet_fails(benchmark, capsys):
    """Two rows for one assay make the construct ambiguous."""
    tmp_path, assays, constructs = benchmark
    rows = [
        {
            "DMS_ID": "Li_2016_tRNA",
            "RNA_TYPE": "tRNA",
            "RAW_CONSTRUCT_SEQ": constructs["Li_2016_tRNA"],
            "Notes": "",
        }
    ] * 2
    write_csv(tmp_path / "reference.csv", REFERENCE_COLUMNS, rows, "\r\n", bom=True)
    status, _ = run(tmp_path, assays)
    assert status == 1
    assert "duplicate DMS_ID" in capsys.readouterr().err


def test_a_missing_assay_table_fails(benchmark, capsys):
    """A corrected assay without a table cannot be migrated."""
    tmp_path, assays, _ = benchmark
    Path(assays / "Li_2016_tRNA.csv").unlink()
    status, _ = run(tmp_path, assays)
    assert status == 1
    assert "not found" in capsys.readouterr().err
