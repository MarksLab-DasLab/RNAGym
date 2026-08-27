"""Score RNA 3D predictions and annotate structural homology."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from fcntl import LOCK_EX, flock
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Tuple

import gemmi
import polars as pl
import RNA_normalizer
from Bio.PDB import PDBIO, MMCIFParser
from RNA_normalizer import mcannotate

from rnagym.config import Config3D
from rnagym.s3d.models import BASELINES, Baseline, homology_columns, prediction_path
from rnagym.s3d.models.utils import valid_structure as valid_prediction
from rnagym.s3d.util import ChainID

# RNA_normalizer still assumes Python 2 and the import-time working directory
RNA_normalizer.xrange = range
mcannotate.MCAnnotate_bin = str(Config3D.MC_ANNOTATE)


@contextmanager
def _cache_lock(path: Path):
    """Serialize creation of one shared cache artifact."""
    lock = path.with_name(f".{path.name}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        flock(handle, LOCK_EX)
        yield


def _ensure_annotation(pdb_path: Path) -> None:
    """Generate one MC-Annotate cache atomically under a file lock."""
    annotation = pdb_path.with_name(f"{pdb_path.name}.mcout")
    with _cache_lock(annotation):
        if (
            annotation.is_file()
            and annotation.stat().st_size
            and annotation.stat().st_mtime >= pdb_path.stat().st_mtime
        ):
            return
        temporary = annotation.with_suffix(".mcout.tmp")
        temporary.unlink(missing_ok=True)
        command = f"{Config3D.MC_ANNOTATE} {pdb_path}"
        with temporary.open("w") as output:
            subprocess.run(shlex.split(command), stdout=output, check=True)
        if not temporary.stat().st_size:
            temporary.unlink()
            raise RuntimeError(f"MC-Annotate failed for {pdb_path}")
        temporary.replace(annotation)


def _interaction_fidelity(reference: list[tuple], prediction: list[tuple]) -> float:
    """Compute INF with the official RNA-Puzzles counting convention."""
    if not reference or not prediction:
        return float(not reference and not prediction)
    # Preserve the official counts without its quadratic nested loops
    # https://github.com/RNA-Puzzles/RNA_assessment/blob/d46f0472e7d52629283bb046a9c8f5d36b35f685/RNA_normalizer/__init__.py#L519-L553
    reference_set = set(reference)
    prediction_set = set(prediction)
    true_positive = sum(item in reference_set for item in prediction)
    false_positive = sum(item not in prediction_set for item in reference)
    false_negative = len(prediction) - true_positive
    precision = true_positive / (true_positive + false_positive)
    sensitivity = true_positive / (true_positive + false_negative)
    return (precision * sensitivity) ** 0.5


@dataclass
class USAlignResult:
    """
    Class for parsing the results of USAlign.

    Members
    -------
    target_name (str):
        The name of the target PDB.
    reference_name (str):
        The name of the reference PDB.
    tm_score (float):
        The TM score of aligning the target to the reference.
    seq_id (float):
        The sequence ID fraction of the target aligned to the reference.
    """

    target_name: str
    ref_name: str
    tm_score: float
    seq_id: float

    __TARGET_NAME_RE = re.compile(r"^Name of Structure_1:\s+(\S+):", re.MULTILINE)
    __REF_NAME_RE = re.compile(r"^Name of Structure_2:\s+(\S+):", re.MULTILINE)
    __TM_SCORE_RE = re.compile(
        r"^TM-score=\s+(\S+)\s+\(normalized by length of Structure_2:", re.MULTILINE
    )
    __ALIGN_RE = re.compile(
        r"^Aligned length=\s+(\S+),\s+RMSD=\s+(\S+),\s+Seq_ID=n_identical/n_aligned=\s+(\S+)",
        re.MULTILINE,
    )

    def __init__(self, usalign_res: str):
        # Names should be stripped of leading "/" and trailing ".pdb"
        target = self.__TARGET_NAME_RE.search(usalign_res)
        self.target_name = os.path.splitext(os.path.basename(target.group(1).strip()))[
            0
        ]
        ref = self.__REF_NAME_RE.search(usalign_res)
        self.ref_name = os.path.splitext(os.path.basename(ref.group(1).strip()))[0]

        tm = self.__TM_SCORE_RE.search(usalign_res)
        self.tm_score = float(tm.group(1))

        align = self.__ALIGN_RE.search(usalign_res)
        self.seq_id = float(align.group(3))


def get_rf2na_chain_ids(pdb_id: str) -> Dict[ChainID, ChainID]:
    """Map input chain IDs to RF2NA output chain IDs in input order."""
    config = Config3D.PREDICTION_DIR / "af3" / "multimers" / pdb_id / "config.json"
    sequences = json.loads(config.read_text())["sequences"]
    mapping = {}
    for index, item in enumerate(sequences):
        chain = next(iter(item.values()))
        mapping[chain["id"]] = chr(ord("A") + index)
    return mapping


def write_chain_minimal_pdb(
    pdb_file: Path, dst_chain_id: str, src_chain_id: str
) -> Path:
    """Extract one chain into a minimal PDB with the requested chain ID."""
    output_file = pdb_file.with_name(f"{pdb_file.stem}_{dst_chain_id}.pdb")
    with _cache_lock(output_file):
        if (
            output_file.is_file()
            and output_file.stat().st_size > 100
            and output_file.stat().st_mtime >= pdb_file.stat().st_mtime
            and valid_prediction(output_file)
        ):
            return output_file

        structure = gemmi.read_structure(str(pdb_file))
        model = structure[0]
        new_structure = gemmi.Structure()
        new_model = gemmi.Model(0)
        new_chain = gemmi.Chain(dst_chain_id)
        old_chain = next(chain for chain in model if chain.name == src_chain_id)
        new_chain.append_residues(old_chain.whole())
        new_model.add_chain(new_chain)
        new_structure.add_model(new_model)
        temporary = output_file.with_suffix(".tmp")
        new_structure.write_minimal_pdb(str(temporary))
        temporary.replace(output_file)
    return output_file


def interaction_network_fidelity(reference_pdb, prediction_pdb):
    """Compare interactions at residues resolved in both structures."""

    def load_struct(pdb_path, idx=None):
        pdb_path = Path(pdb_path)
        _ensure_annotation(pdb_path)
        s = RNA_normalizer.PDBStruct()
        s.load(pdb_path, idx)
        if not s.res_list or not s.raw_sequence():
            raise RuntimeError(f"Failed to parse {pdb_path}")
        return s, s.raw_sequence()

    # 1) Load both fully (no index yet)
    ref_struct, ref_seq = load_struct(reference_pdb)
    pred_struct, pred_seq = load_struct(prediction_pdb)

    # Create a mapping from resid to resname for the predicted structure
    struct = gemmi.read_pdb(str(prediction_pdb))
    resid_to_resname = {
        r.seqid.num: r.name for chain in struct[0].subchains() for r in chain
    }

    # 2) Load the predicted structure with indices matching the reference
    # NOTE(MCA): The reference resids are guaranteed to align with their
    #   position in the sequence.  For example, if the first resid in the
    #   reference structure is 3 it means the first 2 residues from the
    #   sequence are missing
    ref_indices = [
        r.pos
        for r in ref_struct.res_list
        if resid_to_resname.get(r.pos, "N") != "N"
        and r.res.resname in {"A", "C", "G", "U"}
    ]

    with NamedTemporaryFile("w") as tmp:
        ref_chain = ref_struct.res_list[0].chain
        pred_chain = pred_struct.res_list[0].chain

        # Reload reference structure
        tmp.write("\n".join(f"{ref_chain}:{i}:1" for i in ref_indices))
        tmp.flush()
        ref_struct, ref_seq = load_struct(reference_pdb, tmp.name)

        # Reload predicted structure
        tmp.seek(0)
        tmp.truncate()
        tmp.write("\n".join(f"{pred_chain}:{i}:1" for i in ref_indices))
        tmp.flush()
        pred_struct, pred_seq = load_struct(prediction_pdb, tmp.name)

    if ref_seq != pred_seq:
        raise ValueError(f"{ref_seq} != {pred_seq}")

    def inf(interaction_type):
        """Return INF, using 1 if both interaction sets are empty and 0 if only one is."""
        reference = ref_struct.get_interactions(interaction_type)
        prediction = pred_struct.get_interactions(interaction_type)
        return _interaction_fidelity(reference, prediction)

    return inf("PAIR_2D"), inf("PAIR_3D")


class Analysis:
    """Score one prediction against one experimental RNA chain."""

    def __init__(self, pdb_id: str, asym_id: str, auth_id: str, sequence_id: str):
        self.pdb_id = pdb_id.lower()
        self.asym_id = asym_id
        self.auth_id = auth_id
        self.sequence_id = sequence_id

    def baseline_results(
        self, baseline: Baseline, multimer: bool
    ) -> tuple[Path, Path | None]:
        """Locate a prediction and extract its target RNA chain when needed."""
        name = self.pdb_id if multimer else self.sequence_id
        output_dir, prediction = prediction_path(baseline.name, name, multimer)
        if not prediction.is_file():
            return output_dir, None

        if multimer:
            source_chain = self.asym_id
            if baseline.name == "rf2na":
                source_chain = get_rf2na_chain_ids(self.pdb_id)[self.asym_id]
            prediction = write_chain_minimal_pdb(prediction, self.asym_id, source_chain)
        return output_dir, prediction

    def usa_result(self, baseline: Baseline, multimer: bool) -> USAlignResult:
        """Return the US-align result against the experimental structure."""
        _, prediction = self.baseline_results(baseline, multimer)
        if prediction is None:
            raise FileNotFoundError(f"Missing {baseline.name} prediction")
        reference = Config3D.chain_file(self.pdb_id, self.asym_id)
        cache = (
            Config3D.CACHE_DIR
            / "scores"
            / "usalign"
            / baseline.name
            / ("multimers" if multimer else "monomers")
            / f"{self.pdb_id}_{self.asym_id}.out"
        )
        fresh = cache.is_file() and cache.stat().st_mtime >= max(
            prediction.stat().st_mtime, reference.stat().st_mtime
        )
        if fresh:
            result = cache.read_text()
        else:
            command = f"{Config3D.USALIGN_BINARY} {prediction} {reference}"
            result = subprocess.run(
                shlex.split(command), capture_output=True, text=True, check=True
            ).stdout
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(result)
            temporary.replace(cache)
        return USAlignResult(result)

    def inf(self, baseline: Baseline, multimer: bool) -> Tuple[float, float]:
        """Return Watson-Crick and non-Watson-Crick INF scores."""
        _, prediction = self.baseline_results(baseline, multimer)
        if prediction is None:
            raise FileNotFoundError(f"Missing {baseline.name} prediction")
        reference = Config3D.chain_file(self.pdb_id, self.asym_id)

        # Cache model interactions once across repeated experimental structures
        name = f"{self.pdb_id}_{self.asym_id}" if multimer else self.sequence_id
        prediction_pdb = Config3D.CACHE_DIR / "inf" / baseline.name / f"{name}.pdb"
        with _cache_lock(prediction_pdb):
            if (
                not prediction_pdb.is_file()
                or prediction_pdb.stat().st_mtime < prediction.stat().st_mtime
            ):
                prediction_pdb.parent.mkdir(parents=True, exist_ok=True)
                temporary = prediction_pdb.with_suffix(".tmp")
                if prediction.suffix == ".cif":
                    structure = MMCIFParser(QUIET=True).get_structure(name, prediction)
                    for model in structure:
                        for chain in model:
                            chain.id = "A"
                    writer = PDBIO()
                    writer.set_structure(structure)
                    writer.save(str(temporary))
                else:
                    shutil.copy2(prediction, temporary)
                temporary.replace(prediction_pdb)
                annotation = prediction_pdb.with_name(f"{prediction_pdb.name}.mcout")
                annotation.unlink(missing_ok=True)
        return interaction_network_fidelity(reference, prediction_pdb)


def _get_max_tm_results(args):
    """Find each model's closest pre-training structural homolog."""
    index, pdb_id, asym_id, pdb_id_to_date = args
    chain_id = f"{pdb_id.lower()}_{asym_id}"
    source_pdb = Config3D.chain_file(pdb_id, asym_id)
    query_file = _prepare_usalign_pdb(
        source_pdb, Config3D.USALIGN_DIR / "queries" / f"{chain_id}.pdb"
    )
    cached_out_file = Config3D.USALIGN_DIR / f"{chain_id}.out"
    references_file = Config3D.USALIGN_REFERENCES_FILE
    usalign_dir = Config3D.USALIGN_DIR
    usa_cmd = (
        f"{Config3D.USALIGN_BINARY} -dir1 {usalign_dir}/ {references_file} {query_file}"
    )
    # Run US-align if the query or shared reference set changed
    print(f"Processing {chain_id}...")
    fresh = cached_out_file.is_file() and cached_out_file.stat().st_mtime >= max(
        references_file.stat().st_mtime, query_file.stat().st_mtime
    )
    usa_results = cached_out_file.read_text() if fresh else ""
    _, _, last_line = usa_results.rstrip().rpartition("\n")
    if not usa_results or not last_line.startswith("#Total CPU time is"):
        print(f"Running {usa_cmd}")
        result = subprocess.run(
            shlex.split(usa_cmd), capture_output=True, text=True, check=True
        )
        usa_results = result.stdout
        if result.stderr.strip():
            print("---USA Errors---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        temporary = cached_out_file.with_suffix(".out.tmp")
        temporary.write_text(usa_results)
        temporary.replace(cached_out_file)

    raw_results = usa_results.split("\n\n\n")
    try:
        usa_results = [USAlignResult(result) for result in raw_results]
    except Exception as error:
        print(f"`{usa_cmd}` failed with results: {raw_results}")
        raise error

    # Identify the hit with the best TM score
    max_tm_results = {
        b: (0.0, None)  # max_tm, max_tm_result
        for b in BASELINES.values()
    }

    for usa_result in usa_results:
        for bl, (max_tm, max_tm_result) in max_tm_results.items():
            pdb_id, asym_id = usa_result.target_name.split("_")
            published = pdb_id_to_date[pdb_id.lower()]
            if published <= bl.training_cutoff:
                if usa_result.tm_score > max_tm:
                    max_tm_results[bl] = (usa_result.tm_score, usa_result)

    return index, max_tm_results


def _prepare_usalign_pdb(source: Path, output: Path) -> Path:
    """Write one RNA chain in the canonical form expected by US-align."""
    if (
        output.is_file()
        and not output.is_symlink()
        and output.stat().st_mtime >= source.stat().st_mtime
    ):
        return output

    structure = gemmi.read_structure(str(source))
    # US-align ignores modified HETATM records, and its default alignment is sequence independent
    for model in structure:
        for chain in model:
            for residue in chain:
                residue.name = "A"
                residue.het_flag = "A"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pdb.tmp")
    structure.write_minimal_pdb(str(temporary))
    temporary.replace(output)
    return output


def prep_usalign(
    cutoff_col: str = "published", cutoff: str = Config3D.TARGET_CUTOFF
) -> None:
    """Prepare pre-cutoff RNA chains for US-align.

    Parameters
    ----------
    cutoff_col : str
        Column containing ISO publication dates.
    cutoff : str
        Latest publication date included as a reference.
    """
    references_file = Config3D.USALIGN_REFERENCES_FILE
    rna_chains = (
        pl.read_parquet(
            Config3D.ANNOTATED_CHAINS_FILE,
            columns=["pdb_id", "asym_id", cutoff_col],
        )
        .filter(pl.col(cutoff_col) <= cutoff)
        .select("pdb_id", "asym_id")
        .iter_rows()
    )
    Config3D.USALIGN_DIR.mkdir(parents=True, exist_ok=True)
    missing = 0
    updated = False
    names = []
    for pdb_id, asym_id in rna_chains:
        chain_id = f"{pdb_id.lower()}_{asym_id}"
        source_pdb = Config3D.chain_file(pdb_id, asym_id)
        reference_pdb = Config3D.USALIGN_DIR / f"{chain_id}.pdb"
        if not source_pdb.is_file():
            missing += 1
            continue
        previous_mtime = (
            reference_pdb.stat().st_mtime_ns
            if reference_pdb.is_file() and not reference_pdb.is_symlink()
            else None
        )
        _prepare_usalign_pdb(source_pdb, reference_pdb)
        updated |= previous_mtime != reference_pdb.stat().st_mtime_ns
        names.append(f"{chain_id}.pdb")

    contents = "".join(f"{name}\n" for name in names)
    if (
        updated
        or not references_file.is_file()
        or references_file.read_text() != contents
    ):
        temporary = references_file.with_suffix(".txt.tmp")
        temporary.write_text(contents)
        temporary.replace(references_file)
    else:
        print(f"Reusing {references_file}")
    print(f"Prepared {len(names):,} US-align references")
    if missing:
        print(f"Skipped {missing:,} references without cached coordinates")


def add_tm_id(
    chains: pl.DataFrame,
    cutoff_col: str = "published",
    cutoff: str = Config3D.TARGET_CUTOFF,
) -> pl.DataFrame:
    """Annotate chains with their closest pre-training structural homologs.

    Parameters
    ----------
    chains : pl.DataFrame
        Chains identified by ``pdb_id`` and ``asym_id``.
    cutoff_col : str
        Column containing ISO publication dates.
    cutoff : str
        Latest publication date included as a reference.

    Returns
    -------
    pl.DataFrame
        Input rows with model-specific homolog metadata and TM scores.
    """
    references = pl.read_parquet(Config3D.ANNOTATED_CHAINS_FILE).filter(
        pl.col(cutoff_col) <= cutoff
    )
    references_file = Config3D.USALIGN_REFERENCES_FILE
    if not references_file.exists():
        raise RuntimeError(
            "Attempted to call add_tm_id without first calling prep_usalign"
        )

    pdb_id_to_date = {
        pdb_id: published
        for pdb_id, published in references.select("pdb_id", cutoff_col).iter_rows()
    }
    tasks = [
        (index, pdb_id, asym_id, pdb_id_to_date)
        for index, (pdb_id, asym_id) in enumerate(
            chains.select("pdb_id", "asym_id").iter_rows()
        )
    ]
    if len(tasks) == 1:
        results = [_get_max_tm_results(tasks[0])]
    else:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(_get_max_tm_results, tasks))

    reference_by_id = {
        (row["pdb_id"], row["asym_id"]): row
        for row in references.select(
            "pdb_id", "asym_id", "auth_id", "published", "rfam"
        ).iter_rows(named=True)
    }
    rows = chains.to_dicts()
    for index, baseline_results in results:
        for bl, (_, max_tm_result) in baseline_results.items():
            homolog, date, rfam, score = homology_columns(bl.name)
            rows[index].update({homolog: None, date: None, rfam: None, score: 0.0})
            if max_tm_result is None:
                continue
            pdb_id, asym_id = max_tm_result.target_name.split("_", 1)
            match = reference_by_id[(pdb_id.lower(), asym_id)]
            rows[index].update(
                {
                    homolog: f"{pdb_id.lower()}_{match['auth_id']}",
                    date: match["published"],
                    rfam: match["rfam"],
                    score: max_tm_result.tm_score,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)
