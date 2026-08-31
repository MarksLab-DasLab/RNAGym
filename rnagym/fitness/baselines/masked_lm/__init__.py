"""
Shared masked-marginal scoring for the benchmark's masked RNA language models.

The four fill strategies live in ``strategies``, batched inference in
``engine``, the model interface in ``adapter``, and the command line entry
point in ``runner``. A model's scoring script is an adapter plus a call to
``runner.main``.
"""

from rnagym.fitness.baselines.masked_lm.adapter import MaskedLMAdapter
from rnagym.fitness.baselines.masked_lm.engine import accumulate_scores, window_contexts
from rnagym.fitness.baselines.masked_lm.runner import main
from rnagym.fitness.baselines.masked_lm.strategies import (
    MASK_CHAR,
    STRATEGIES,
    TaskTable,
    build_tasks,
    normalize_strategy,
    parse_mutations,
    recover_wild_type,
)

__all__ = [
    "MASK_CHAR",
    "STRATEGIES",
    "MaskedLMAdapter",
    "TaskTable",
    "accumulate_scores",
    "build_tasks",
    "main",
    "normalize_strategy",
    "parse_mutations",
    "recover_wild_type",
    "window_contexts",
]
