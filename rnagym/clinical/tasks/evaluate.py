"""Evaluate merged clinical scores: AUROC and average precision per group.

Pathogenicity is the negated model score (a less likely alternate allele is
more pathogenic). Every model in a comparison is evaluated on the same rows:
those with a finite score from all of them. A group is reportable with at
least ``MIN_EACH`` pathogenic and benign rows; ``unknown`` never is.

Groups. Nuclear files: every consequence group (a pre-mRNA row in two groups
counts in both), the intron distance bins, and for non-SNV files the variant
class. Mito files: every ``mt_stratum``, with one benign set, every label-0 row
(the labels already encode the expert panel's call or gnomAD frequency).

Headlines per nuclear file and cohort: the consequence macro (mean AUROC over
the reportable consequence groups) and the gene macros (mean AUROC over genes
with ``MIN_EACH`` of each class, among all rows and among missense rows).
Intervals come from a cluster bootstrap, by gene (mito: by processed RNA unit),
with the same draws for every model so that differences are paired.

Baselines. Nuclear: the out-of-fold pathogenic rate of the row's transcript
consequence (five gene-grouped folds, Laplace smoothing). Mito: the frequency
used by the benign rule, negated, on the rows where it is known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import average_precision_score

from rnagym.clinical.models import MODELS
from rnagym.clinical.tasks.prepare import MITO, NUCLEAR
from rnagym.config import ConfigClinical

MIN_EACH = 10
NUCLEAR_META = (
    "row_id",
    "binary_label",
    "cohort",
    "gene",
    "variant_class",
    "consequence_groups",
    "consequence_tx",
    "intron_distance_bin",
)
MITO_META = (
    "row_id",
    "binary_label",
    "cohort",
    "gene",
    "variant_class",
    "mt_stratum",
    "benign_rule_freq",
    "unit",
)
# The processed RNAs of the 37 mitochondrial genes, the bootstrap clusters of the
# mito files: 22 tRNAs, 2 rRNAs and 13 mRNAs, two of them bicistronic (35 units)
MITO_UNITS = frozenset(
    {"MT-T" + x for x in "A C D E F G H I K L1 L2 M N P Q R S1 S2 T V W Y".split()}
    | {"MT-RNR1", "MT-RNR2"}
    | {"MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8/ATP6", "MT-CO3", "MT-ND3"}
    | {"MT-ND4L/ND4", "MT-ND5", "MT-ND6", "MT-CYB"}
)
# Every group record carries these columns, null when the group lacks a class
METRICS = (
    "prevalence",
    "auroc",
    "average_precision",
    "auroc_ci_low",
    "auroc_ci_high",
    "bootstrap_invalid",
)


def auroc_weighted(
    inverse: npt.NDArray[np.intp],
    n_unique: int,
    label: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
) -> float:
    """Return the AUROC with row weights, from scores already reduced to tie ranks."""
    wp = np.bincount(inverse, weights=weight * label, minlength=n_unique)
    wb = np.bincount(inverse, weights=weight * (1 - label), minlength=n_unique)
    total_p, total_b = wp.sum(), wb.sum()
    if total_p == 0 or total_b == 0:
        return np.nan
    below = np.concatenate(([0.0], np.cumsum(wb)[:-1]))
    return float((wp * (below + 0.5 * wb)).sum() / (total_p * total_b))


def auroc(score: npt.NDArray[np.float64], label: npt.NDArray[np.float64]) -> float:
    """Return the AUROC of a score, ties counted as half."""
    uniq, inverse = np.unique(score, return_inverse=True)
    return auroc_weighted(inverse, len(uniq), label, np.ones(len(label)))


def consequence_baseline(meta: pl.DataFrame) -> npt.NDArray[np.float64]:
    """Return the out-of-fold pathogenic rate of each row's transcript consequence."""
    genes = meta["gene"].to_list()
    fold = np.array([zlib.crc32(g.encode()) % 5 for g in genes])
    label = meta["binary_label"].to_numpy()
    cons = meta["consequence_tx"].fill_null("").to_list()
    out = np.empty(len(meta))
    for k in range(5):
        train, test = fold != k, fold == k
        rate = (label[train].sum() + 1) / (train.sum() + 2)
        table: dict[str, list[int]] = {}
        for c, y in zip(np.array(cons, dtype=object)[train], label[train]):
            entry = table.setdefault(c, [0, 0])
            entry[0] += int(y)
            entry[1] += 1
        for i in np.flatnonzero(test):
            p, n = table.get(cons[i], (0, 0))
            out[i] = (p + 1) / (n + 2) if n else rate
    return out


def mito_clusters(meta: pl.DataFrame) -> npt.NDArray[np.object_]:
    """Return the processed RNA unit of every mito row, its bootstrap cluster."""
    units = meta["unit"].fill_null("").to_numpy()
    unknown = set(units) - MITO_UNITS
    if unknown:
        raise ValueError(f"rows outside the 35 processed RNA units: {sorted(unknown)}")
    return units


def groups_for(
    meta: pl.DataFrame, name: str
) -> list[tuple[str, str, npt.NDArray[np.bool_]]]:
    """Return (group type, group, row mask) for every group of a file."""
    n = len(meta)
    if name.startswith("mt_"):
        strata = ["all"] + sorted(set(meta["mt_stratum"].to_list()))
        out = []
        for stratum in strata:
            in_stratum = (
                np.ones(n, dtype=bool)
                if stratum == "all"
                else meta["mt_stratum"].to_numpy() == stratum
            )
            out.append(("mito:all_benign", stratum, in_stratum))
        return out
    out = [("all", "all", np.ones(n, dtype=bool))]
    groups = meta["consequence_groups"].to_list()
    for g in sorted({x for row in groups for x in row.split("|")}):
        out.append(
            ("consequence", g, np.array([g in row.split("|") for row in groups]))
        )
    bins = meta["intron_distance_bin"].to_numpy()
    for b in sorted(set(bins) - {""}):
        out.append(("intron_bin", b, bins == b))
    if not name.endswith("_snv"):
        classes = meta["variant_class"].to_numpy()
        for c in sorted(set(classes)):
            out.append(("variant_class", c, classes == c))
    return out


def records_frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Return a table from records, with the schema read from every record."""
    return pl.DataFrame(records, infer_schema_length=None)


def group_record(
    name: str,
    cohort: str,
    row_set: str,
    kind: str,
    group: str,
    pred: str,
    reportable: bool,
    values: npt.NDArray[np.float64],
    label: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    weights: list[npt.NDArray[np.float64]],
    draws: int,
) -> dict[str, Any]:
    """Return the AUROC, with a paired cluster-bootstrap interval, and AP of one group."""
    n_p, n_b = int(label[mask].sum()), int((1 - label[mask]).sum())
    record = {
        "name": name,
        "cohort": cohort,
        "row_set": row_set,
        "group_type": kind,
        "group": group,
        "predictor": pred,
        "n_pathogenic": n_p,
        "n_benign": n_b,
        "reportable": reportable,
        **dict.fromkeys(METRICS),
    }
    if n_p and n_b:
        uniq, inverse = np.unique(values[mask], return_inverse=True)
        boot = np.array(
            [
                auroc_weighted(inverse, len(uniq), label[mask], w[mask])
                for w in weights[1:]
            ]
        )
        point = auroc_weighted(inverse, len(uniq), label[mask], weights[0][mask])
        record |= {
            "prevalence": n_p / (n_p + n_b),
            "auroc": point,
            "average_precision": float(
                average_precision_score(label[mask], values[mask])
            ),
            "auroc_ci_low": float(np.nanpercentile(boot, 2.5)) if draws else np.nan,
            "auroc_ci_high": float(np.nanpercentile(boot, 97.5)) if draws else np.nan,
            "bootstrap_invalid": int(np.isnan(boot).sum()),
        }
    return record


def evaluate(
    name: str, staged: Path, merged: Path, models: Sequence[str], draws: int, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Return the group table and, for nuclear files, the headline table of a file."""
    mito = name.startswith("mt_")
    columns = list(MITO_META if mito else NUCLEAR_META)
    meta = pl.read_parquet(staged / f"{name}.parquet", columns=columns)
    meta = meta.with_columns(pl.col("binary_label").cast(pl.Int8))
    scores = {}
    for model in models:
        frame = pl.read_parquet(
            merged / name / f"{model}.parquet", columns=["row_id", "score"]
        )
        joined = meta.select("row_id").join(frame, on="row_id", how="left")
        scores[model] = joined["score"].to_numpy()
    finite = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    baselines = {}
    if mito:
        freq = meta["benign_rule_freq"].cast(pl.Float64, strict=False).to_numpy()
        baselines["baseline_frequency"] = -freq
    else:
        baselines["baseline_consequence"] = consequence_baseline(meta)

    rows: list[dict[str, Any]] = []
    headline: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for cohort in sorted(set(meta["cohort"].to_list())):
        keep = finite & (meta["cohort"].to_numpy() == cohort)
        sub = meta.filter(pl.Series(keep))
        label = sub["binary_label"].to_numpy().astype(float)
        clusters = mito_clusters(sub) if mito else sub["gene"].to_numpy()
        cluster_ids, cluster_of_row = np.unique(clusters, return_inverse=True)
        # Draw 0 is the point estimate (every cluster once); draws 1.. resample clusters
        n_clusters = len(cluster_ids)
        cluster_draws = [np.ones(n_clusters)] + [
            np.bincount(
                rng.integers(0, n_clusters, n_clusters), minlength=n_clusters
            ).astype(float)
            for _ in range(draws)
        ]
        weights = [c[cluster_of_row] for c in cluster_draws]
        predictors = {m: -v[keep] for m, v in scores.items()} | {
            b: v[keep] for b, v in baselines.items()
        }
        # Row sets: every model-scored row, and the complete cases where the baseline
        # is also known; models and baseline are compared only on the latter
        baseline_known = np.all(
            [np.isfinite(v[keep]) for v in baselines.values()], axis=0
        )
        row_sets = {
            "model_rows": (np.ones(len(sub), dtype=bool), list(scores)),
            "complete_cases": (baseline_known, list(predictors)),
        }
        reportable_consequence = []
        for kind, group, group_mask in groups_for(sub, name):
            for row_set, (in_set, preds) in row_sets.items():
                mask = group_mask & in_set
                n_p, n_b = int(label[mask].sum()), int((1 - label[mask]).sum())
                reportable = n_p >= MIN_EACH and n_b >= MIN_EACH and group != "unknown"
                if kind == "consequence" and reportable and row_set == "model_rows":
                    reportable_consequence.append(mask)
                for pred in preds:
                    rows.append(
                        group_record(
                            name,
                            cohort,
                            row_set,
                            kind,
                            group,
                            pred,
                            reportable,
                            predictors[pred],
                            label,
                            mask,
                            weights,
                            draws,
                        )
                    )
        if mito:
            continue
        headline += headline_records(
            name,
            cohort,
            sub,
            label,
            keep,
            scores,
            predictors,
            reportable_consequence,
            cluster_ids,
            cluster_draws,
            weights,
            draws,
        )
    return records_frame(rows), records_frame(headline) if headline else None


def headline_records(
    name: str,
    cohort: str,
    sub: pl.DataFrame,
    label: npt.NDArray[np.float64],
    keep: npt.NDArray[np.bool_],
    scores: dict[str, npt.NDArray[np.float64]],
    predictors: dict[str, npt.NDArray[np.float64]],
    reportable_consequence: list[npt.NDArray[np.bool_]],
    cluster_ids: npt.NDArray[np.object_],
    cluster_draws: list[npt.NDArray[np.float64]],
    weights: list[npt.NDArray[np.float64]],
    draws: int,
) -> list[dict[str, Any]]:
    """Return each model's consequence and gene macros with intervals for one cohort."""
    genes = sub["gene"].to_numpy()
    missense = np.array(
        [
            "missense_variant" in g.split("|")
            for g in sub["consequence_groups"].to_list()
        ]
    )

    def qualifying(mask: npt.NDArray[np.bool_]) -> list[str]:
        return [
            g
            for g in np.unique(genes[mask])
            if label[mask & (genes == g)].sum() >= MIN_EACH
            and (1 - label[mask & (genes == g)]).sum() >= MIN_EACH
        ]

    everything = np.ones(len(sub), dtype=bool)
    gene_sets = {
        "all": (everything, qualifying(everything)),
        "missense": (missense, qualifying(missense)),
    }
    cluster_index = {g: k for k, g in enumerate(cluster_ids)}
    records = []
    for pred in scores:  # baselines are not ranked on the headline
        values = predictors[pred]
        # Sorting and per-gene AUROCs do not change between draws: compute them
        # once. A zero-weight row contributes nothing, so weighting the full group
        # is the same as dropping the clusters a draw leaves out
        inverses = [
            (np.unique(values[m], return_inverse=True)[1], m)
            for m in reportable_consequence
        ]
        ranked = [(inv, int(inv.max()) + 1, m) for inv, m in inverses]
        gene_auc = {
            scope: (
                np.array(
                    [
                        auroc(values[mask & (genes == g)], label[mask & (genes == g)])
                        for g in kept
                    ]
                ),
                np.array([cluster_index[g] for g in kept], dtype=np.int64),
            )
            for scope, (mask, kept) in gene_sets.items()
        }

        def consequence_macro(k: int) -> float:
            w = weights[k]
            vals = [auroc_weighted(inv, n, label[m], w[m]) for inv, n, m in ranked]
            return float(np.mean(vals)) if vals and not np.isnan(vals).any() else np.nan

        def gene_macro(scope: str, k: int) -> float:
            aucs, idx = gene_auc[scope]
            mult = cluster_draws[k][idx]
            if len(aucs) and mult.sum() > 0:
                return float(np.average(aucs, weights=mult))
            return np.nan

        record: dict[str, Any] = {
            "name": name,
            "cohort": cohort,
            "predictor": pred,
            "rows_scored_by_all_models": int(keep.sum()),
            "consequence_groups": len(reportable_consequence),
        }
        macros: tuple[tuple[str, Callable[[int], float]], ...] = (
            ("consequence_macro", consequence_macro),
            ("gene_macro", lambda k: gene_macro("all", k)),
            ("missense_gene_macro", lambda k: gene_macro("missense", k)),
        )
        for label_name, fn in macros:
            point = fn(0)
            boot = np.array([fn(k) for k in range(1, draws + 1)])
            low = float(np.nanpercentile(boot, 2.5)) if draws else np.nan
            high = float(np.nanpercentile(boot, 97.5)) if draws else np.nan
            record[f"{label_name}_auroc"] = point
            record[f"{label_name}_ci_low"] = low
            record[f"{label_name}_ci_high"] = high
            record[f"{label_name}_invalid_draws"] = int(np.isnan(boot).sum())
        record["genes"] = len(gene_sets["all"][1])
        record["missense_genes"] = len(gene_sets["missense"][1])
        records.append(record)
    return records


def write_settings(out: Path, name: str, args: argparse.Namespace) -> None:
    """Write one file's settings, with the evaluator's sha256, to <name>.settings.json."""
    settings = {
        **vars(args),
        "names": [name],
        "min_each": MIN_EACH,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    text = json.dumps(settings, default=str, indent=2)
    (out / f"{name}.settings.json").write_text(text)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--staged", type=Path, default=ConfigClinical.STAGED_DIR)
    parser.add_argument(
        "--merged",
        type=Path,
        default=ConfigClinical.MERGED_DIR,
        help="merged/<name>/<model>.parquet",
    )
    parser.add_argument("--names", nargs="+", default=[*NUCLEAR, *MITO])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--out", type=Path, default=ConfigClinical.EVALUATION_DIR)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261008)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate each named file into its group, headline and settings files."""
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        table, headline = evaluate(
            name, args.staged, args.merged, args.models, args.bootstrap, args.seed
        )
        table.write_csv(args.out / f"{name}.groups.csv")
        if headline is not None:
            headline.write_csv(args.out / f"{name}.headline.csv")
        write_settings(args.out, name, args)
        print(name, len(table), flush=True)


if __name__ == "__main__":
    main()
