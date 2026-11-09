"""Zero-shot scoring of the ClinVar and MITOMAP clinical variant benchmark.

``tasks.prepare`` stages the released files once. ``score_masked`` and
``score_evo2`` score staged rows with masked and autoregressive models,
``tasks.merge`` checks shard coverage, ``tasks.evaluate`` computes AUROC within
consequence classes and ``tasks.leaderboard`` writes the leaderboard.
"""
