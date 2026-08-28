"""Prediction folders and score columns used by the fitness workflow."""

ALL_MODELS = (
    "evo1",
    "evo1.5",
    "evo2",
    "evo2_40b",
    "GenSLM",
    "NT",
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

SCORE_COLS = {
    "evo1": "evo_1_131k_base_score",
    "evo1.5": "evo_1.5_8k_base_score",
    "evo2": "evo2_7b_score",
    "evo2_40b": "evo2_40b_score",
    "GenSLM": "logit_scores",
    "NT": "kmer_pseudo_LL",
    "RNA-FM": {"folder": "rna_fm_4fill", "column": "RNA_FM_scores_wt_fill"},
    "rinalmo": {"folder": "rinalmo_4fill", "column": "logit_scores_wt_fill"},
    "RNAErnie": "Mutation_Scores",
    "orthrus": {"folder": "orthrus_4fill", "column": "orthrus_score_wt_fill"},
    "aido_rna": {"folder": "aido_rna_4fill", "column": "aido_rna_score_wt_fill"},
    "rnagenesis": {
        "folder": "rnagenesis_4fill",
        "column": "rnagenesis_score_wt_fill",
    },
    "aido_rna_1m": {
        "folder": "aido_rna_1m_4fill",
        "column": "aido_rna_score_wt_fill",
    },
    "aido_rna_25m": {
        "folder": "aido_rna_25m_4fill",
        "column": "aido_rna_score_wt_fill",
    },
    "aido_rna_300m": {
        "folder": "aido_rna_300m_4fill",
        "column": "aido_rna_score_wt_fill",
    },
    "aido_rna_650m": {
        "folder": "aido_rna_650m_4fill",
        "column": "aido_rna_score_wt_fill",
    },
    "EVmutation": "prediction_epistatic",
}

FOUR_FILL_SPECS = {
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


def resolve_source(score_columns: dict, model_name: str) -> tuple[str, str]:
    """Return the prediction folder and score column for a model entry."""
    try:
        specification = score_columns[model_name]
    except KeyError:
        raise KeyError(f"No score column configured for model {model_name!r}") from None
    if isinstance(specification, str):
        return model_name, specification
    return specification["folder"], specification["column"]
