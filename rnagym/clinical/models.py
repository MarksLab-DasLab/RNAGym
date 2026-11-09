"""Checkpoints scored on the clinical benchmark, with their context caps.

``cap`` is the context in nucleotides: each model's own position limit after
special tokens, or the benchmark ceiling of 10,101 (the pre-mRNA window) for
models without one. Local weights are read where the fitness benchmark's
``fetch-weight.sh`` writes them; Hub checkpoints use the pinned revisions of
``model_registry``.
"""

from __future__ import annotations

from pathlib import Path

from rnagym.config import ConfigClinical

CEILING = 10_101

_CHECKPOINTS = ConfigClinical.CHECKPOINT_DIR
_BASELINES = "rnagym.fitness.baselines"
_AIDO = (f"{_BASELINES}.AIDO_RNA.score_aido_rna_single_dms", "AIDORNAAdapter")
_NTV3 = (f"{_BASELINES}.Nucleotide_Transformer.compute_fitness", "NTv3Adapter")

# key: (module, adapter class, load arguments, cap, max batch tokens or None). Load
# arguments keep the types the adapters expect: RNAGenesis joins paths to its
# directory
MASKED: dict[str, tuple[str, str, dict[str, str | Path], int, int | None]] = {
    "rna_fm": (
        f"{_BASELINES}.RNA_FM.score_rna_fm_single_dms",
        "RNAFMAdapter",
        {
            "checkpoint_path": str(
                _CHECKPOINTS / "rna-fm/hub/checkpoints/RNA-FM_pretrained.pth"
            )
        },
        1_022,
        None,
    ),
    "rinalmo": (
        f"{_BASELINES}.RiNALMo.score_rinalmo_single_dms",
        "RiNALMoAdapter",
        {"checkpoint_path": str(_CHECKPOINTS / "rinalmo/rinalmo_giga_pretrained.pt")},
        1_022,
        None,
    ),
    "rnagenesis": (
        f"{_BASELINES}.RNAGenesis.score_rnagenesis_single_dms",
        "RNAGenesisAdapter",
        {"model_name": _CHECKPOINTS / "rnagenesis"},
        2_048,  # config seq_length; the adapter adds no special tokens
        None,
    ),
    "aido_rna_1m": (
        *_AIDO,
        {"model_name": "genbio-ai/AIDO.RNA-1M-MARS"},
        1_022,
        221_184,
    ),
    "aido_rna_25m": (
        *_AIDO,
        {"model_name": "genbio-ai/AIDO.RNA-25M-MARS"},
        1_022,
        221_184,
    ),
    "aido_rna_300m": (
        *_AIDO,
        {"model_name": "genbio-ai/AIDO.RNA-300M-MARS"},
        1_022,
        147_456,
    ),
    "aido_rna_650m": (
        *_AIDO,
        {"model_name": "genbio-ai/AIDO.RNA-650M"},
        1_022,
        98_304,
    ),
    "aido_rna": (*_AIDO, {"model_name": "genbio-ai/AIDO.RNA-1.6B"}, 1_022, 49_152),
    "ntv3_8m": (*_NTV3, {"model_name": "InstaDeepAI/NTv3_8M_pre"}, CEILING, None),
    "ntv3_100m": (*_NTV3, {"model_name": "InstaDeepAI/NTv3_100M_pre"}, CEILING, None),
    "ntv3_650m": (*_NTV3, {"model_name": "InstaDeepAI/NTv3_650M_pre"}, CEILING, None),
    "orthrus": (
        f"{_BASELINES}.Orthrus.score_orthrus_single_dms",
        "OrthrusAdapter",
        {"model_name": "antichronology/orthrus-mlm-6-track"},
        CEILING,
        None,
    ),
}

# key: (cap, tokens per batch). Evo 2 scores are per-token means, small numbers,
# and FP8 results depend on batch shape: windows batched together moved scores
# by up to 0.009 against the same window scored alone. So every window is scored
# alone (a budget of 1 token gives one window per batch).
EVO2: dict[str, tuple[int, int]] = {
    "evo2_1b_base": (8_192, 1),
    "evo2_7b": (CEILING, 1),
    "evo2_20b": (CEILING, 1),
    "evo2_40b": (CEILING, 1),
}

# Every scored checkpoint, in the order of the published evaluation
MODELS = (*MASKED, *EVO2)
