"""Assay groups, checkpoint revisions and prediction columns for fitness workflows."""

from __future__ import annotations

from pathlib import Path

CHECKPOINT_MODELS = (
    "evo1",
    "evo1.5",
    "evo2",
    "evo2_1b_base",
    "evo2_20b",
    "evo2_40b",
    "GenSLM",
    "ntv3_8m",
    "ntv3_100m",
    "ntv3_650m",
    "rinalmo",
    "RNAErnie",
    "RNA-FM",
    "orthrus",
    "aido_rna",
    "rnagenesis",
    "aido_rna_1m",
    "aido_rna_25m",
    "aido_rna_300m",
    "aido_rna_650m",
)
ALL_MODELS = (*CHECKPOINT_MODELS, "EVmutation")

ASSAY_GROUPS = {
    "ncRNA": ("Ribozyme", "tRNA", "Aptamer"),
    "non-coding": ("Ribozyme", "tRNA", "Aptamer", "mRNA-splicing"),
    "coding": ("mRNA-coding",),
}

CHECKPOINT_REVISIONS = {
    "InstaDeepAI/NTv3_8M_pre": "c57a813117f0f90142098f81cc912b3357c9ecd1",
    "InstaDeepAI/NTv3_100M_pre": "5c685dca15891f5c5b80e0c930e23b87a217e441",
    "InstaDeepAI/NTv3_650M_pre": "5e6050bed864a5a8fb32481096bf555495316b31",
    "antichronology/orthrus-mlm-6-track": "5f0dc87d51065035fc28e71972c69f9c84f4deae",
    "arcinstitute/evo2_1b_base": "2279e1df422c991037470302360edd40d0d2ea1e",
    "arcinstitute/evo2_7b": "bda0089f92582d5baabf0f22d9fc85f3588f6b58",
    "arcinstitute/evo2_20b": "8b0f0a9a70c66367ed181a17d049b95699a28fed",
    "arcinstitute/evo2_40b": "d529aa57c30771814217ad89baaeaf6e2315c7d7",
    "evo-design/evo-1.5-8k-base": "99a9a4df722662b03d2a79b1770c3150421aa9e9",
    "genbio-ai/AIDO.RNA-1M-MARS": "00029ce0f63d5c40ef6cc12ddd7744dcc5a97a42",
    "genbio-ai/AIDO.RNA-25M-MARS": "f7f0bfaeabf4fbd258995dab82c5cf356ec6e852",
    "genbio-ai/AIDO.RNA-300M-MARS": "0f21db50d971d43ee66b39963ae4ec3c28b66e0d",
    "genbio-ai/AIDO.RNA-650M": "efaf4af45dc6b522b7ba62ee725d00ad9c43ce11",
    "genbio-ai/AIDO.RNA-1.6B": "cea90e53284c2a8b77c66528303d04a784b9c67b",
    "togethercomputer/evo-1-131k-base": "c206aab77ae5967a069c4200ecb1858588528c9d",
}

SCORE_COLS: dict[str, str | dict[str, str]] = {
    "evo1": "evo_1_131k_base_score",
    "evo1.5": "evo_1.5_8k_base_score",
    "evo2": "evo2_7b_score",
    "evo2_1b_base": "evo2_1b_base_score",
    "evo2_20b": "evo2_20b_score",
    "evo2_40b": "evo2_40b_score",
    "GenSLM": "logit_scores",
    "RNAErnie": "Mutation_Scores",
    "EVmutation": "prediction_epistatic",
}

FOUR_FILL_SPECS = {
    "ntv3_8m": ("ntv3_8m_4fill", "ntv3_score"),
    "ntv3_100m": ("ntv3_100m_4fill", "ntv3_score"),
    "ntv3_650m": ("ntv3_650m_4fill", "ntv3_score"),
    "rna_fm": ("rna_fm_4fill", "RNA_FM_scores"),
    "rinalmo": ("rinalmo_4fill", "logit_scores"),
    "rnagenesis": ("rnagenesis_4fill", "rnagenesis_score"),
    "aido_rna": ("aido_rna_4fill", "aido_rna_score"),
    "aido_rna_1m": ("aido_rna_1m_4fill", "aido_rna_score"),
    "aido_rna_25m": ("aido_rna_25m_4fill", "aido_rna_score"),
    "aido_rna_300m": ("aido_rna_300m_4fill", "aido_rna_score"),
    "aido_rna_650m": ("aido_rna_650m_4fill", "aido_rna_score"),
    "orthrus": ("orthrus_4fill", "orthrus_score"),
}
STRATEGIES = ("wt_fill", "mask_fill", "mut_fill", "match_fill")

SCORE_COLS.update(
    {
        ("RNA-FM" if name == "rna_fm" else name): {
            "folder": folder,
            "column": f"{stem}_wt_fill",
        }
        for name, (folder, stem) in FOUR_FILL_SPECS.items()
    }
)

_four_fill_models = []
for name, (folder, column_stem) in FOUR_FILL_SPECS.items():
    entries = {
        f"{name}_{strategy}": {
            "folder": folder,
            "column": f"{column_stem}_{strategy}",
        }
        for strategy in STRATEGIES
    }
    SCORE_COLS.update(entries)
    _four_fill_models.extend(entries)
FOUR_FILL_MODELS = tuple(_four_fill_models)


def checkpoint_revision(model: str) -> str | None:
    """Resolve a pinned Hub checkpoint or an explicit local model directory."""
    if Path(model).is_dir():
        return None
    try:
        return CHECKPOINT_REVISIONS[model]
    except KeyError:
        raise ValueError(
            f"Unpinned checkpoint: {model!r}. "
            "Use a registered checkpoint or a local model directory."
        ) from None


def resolve_source(
    score_columns: dict[str, str | dict[str, str]], model_name: str
) -> tuple[str, str]:
    """Return the prediction folder and score column for a model entry."""
    try:
        specification = score_columns[model_name]
    except KeyError:
        raise KeyError(f"No score column configured for model {model_name!r}") from None
    if isinstance(specification, str):
        return model_name, specification
    return specification["folder"], specification["column"]
