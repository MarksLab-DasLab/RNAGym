# RNAGym Fitness Leaderboard

**Metric (updated in v0.1.1):** signed Spearman correlation between each model's score and the
measured DMS fitness, macro-averaged over the 3 ncRNA categories (ribozyme, tRNA, aptamer), with
each category weighted equally. This replaces the previous metric (absolute Spearman averaged over
all assays, including mRNA coding and splicing).

Two changes from before:
1. Signed, not absolute: a negative value means the model's score is negatively correlated with fitness.
2. ncRNA only: the mRNA coding and splicing assays are excluded (see rationale below).

## Leaderboard: signed Spearman, macro-mean over 3 ncRNA categories

| Rank | Model | Ribozyme (n=26) | tRNA (n=3) | Aptamer (n=2) | Macro (3 ncRNA) |
|---:|:--|--:|--:|--:|--:|
| 1 | AIDO.RNA (650M) | 0.0660 | 0.4894 | 0.0934 | 0.2163 |
| 2 | Evo 2 (40B) | 0.1081 | 0.4310 | 0.0970 | 0.2120 |
| 3 | AIDO.RNA (1.6B) | 0.0609 | 0.4884 | 0.0718 | 0.2070 |
| 4 | RNA-ERNIE | 0.1343 | 0.4161 | 0.0306 | 0.1937 |
| 5 | Evo 2 (7B) | 0.0651 | 0.3867 | 0.1192 | 0.1904 |
| 6 | RNAGenesis | 0.0750 | 0.4381 | 0.0343 | 0.1825 |
| 7 | RiNALMo | -0.0243 | 0.4856 | 0.0459 | 0.1690 |
| 8 | AIDO.RNA (300M) | -0.0256 | 0.4551 | 0.0372 | 0.1556 |
| 9 | AIDO.RNA (25M) | -0.0299 | 0.4569 | 0.0348 | 0.1540 |
| 10 | Evo 1.5 | 0.0278 | 0.3850 | 0.0007 | 0.1378 |
| 11 | RNA-FM | -0.0225 | 0.4147 | -0.0043 | 0.1293 |
| 12 | Nucleotide Transformer | 0.1329 | 0.3166 | -0.0886 | 0.1203 |
| 13 | AIDO.RNA (1M) | 0.0014 | 0.1777 | 0.0626 | 0.0806 |
| 14 | Orthrus | -0.0349 | 0.0922 | 0.1578 | 0.0717 |
| 15 | Evo 1 | -0.0216 | 0.0948 | 0.0058 | 0.0263 |
| 16 | GenSLM | -0.0045 | -0.0934 | -0.0036 | -0.0338 |

AIDO.RNA (650M), Evo 2 (40B) and AIDO.RNA (1.6B) span 0.0093. With only 31 assays, this difference
should not be treated as a resolved ordering.

All five released AIDO.RNA checkpoints are listed as separate entries rather than in a side table,
since they are separate models scored the same way. Their scores rise steeply to 650M and then stop:
the 650M scores above the 1.6B under `wt-fill` and `match-fill`, though not under `mask-fill` or
`mut-fill`, so the shape of that curve depends on the scoring convention and should be read with that
in mind.

Per category columns are the mean signed Spearman over the assays in that category. Macro is the
unweighted average of the three category means. Underlying values:
[`leaderboard_signed_3ncRNA.csv`][0].

The masked language models other than RNA-ERNIE are scored with `wt-fill`, the convention used by
the ESM and ProteinGym reference implementations. RNA-ERNIE retains its original scorer: one
unmasked wild-type forward pass followed by a sum of raw probability differences at mutated
positions. It is a separate zero-shot method, not a masked-marginal score. Orthrus uses its official
[MLM variant-scoring interface][1].
Its optional CDS and splice channels stay zero because the DMS assays do not provide those
annotations.

## Scoring convention

The masked language models are scored with the masked-marginal log-likelihood ratio: mask a mutated
position, take `log p(mutant) - log p(wild type)` there, and sum over the variant's mutated
positions. Meier et al. 2021 (ESM-1v), supplement Appendix A defines three context choices. A fourth,
closely related convention is implemented by the authors' released code. They differ in exactly one
thing: **what fills the variant's OTHER mutated positions while one position is masked.** With `M`
the mutated positions, `x_-i` a mask at `i` and `x_-M` masks at every position in `M`:

| name | fill at the other mutated sites | score |
|:--|:--|:--|
| **`wt-fill`** | wild-type bases | `sum_i log p(mt_i \| x^wt_-i) - log p(wt_i \| x^wt_-i)` |
| `mask-fill` | masks | `sum_i log p(mt_i \| x^wt_-M) - log p(wt_i \| x^wt_-M)` |
| `mut-fill` | mutant bases | `sum_i log p(mt_i \| x^mt_-i) - log p(wt_i \| x^mt_-i)` |
| `match-fill` | the allele being scored | `sum_i log p(mt_i \| x^mt_-i) - log p(wt_i \| x^wt_-i)` |

**The leaderboard uses `wt-fill`.** It is what the ESM authors' own
[ESM example][2] and [ProteinGym baseline][3]
implement under the option name `masked-marginals`, so it is the convention the surrounding zero-shot
literature is calibrated on. The choice is on that provenance, not on which scores best.

All four are computed and published, see the sensitivity section below. `fitness/baselines/masked_lm`
computes them in a single pass per checkpoint: the four share most of their masked contexts, so all
four together cost about 19% more unique context examples than `mut-fill` alone (2,929,196 against
2,458,521 over the 31 assays). These are batched inputs, not model invocations. Every scoring script
accepts `--strategies`.

Two properties worth knowing. **On single mutants all four are identical**, because a variant with
one mutation has no other mutated positions. That is used as a fixture test. And **99.4% of the ncRNA
variants are multi-mutants**, so the conventions diverge on essentially everything here.

The name is overloaded and the overloading has caused real errors. The ESM code option called
`masked-marginals` is `wt-fill`, while the formula written in the ESM paper is `mask-fill`, and
`wt-marginals` is a different method again: one unmasked forward pass with no masking at all.

RiNALMo and RNA-FM were also rescored to fix outright bugs, not just the convention. The earlier
RiNALMo scores looked the mutant base up in an RNA-form sequence while its alphabet is DNA, so
mutations to U scored against the unknown-token logit. The earlier RNA-FM scores match a single
unmasked wild-type pass (`wt-marginals`) rather than the masked strategy its run script requested.

## Sensitivity to the fill strategy

Every masked model is scored under all four conventions, so the effect of the choice is visible
rather than assumed. Macro over the 3 ncRNA categories. Regenerate this table, the per-category
values and the category spreads with `fitness/analyze_fill_strategies.py`.

| Checkpoint | `wt-fill` | `mask-fill` | `mut-fill` | `match-fill` |
|:--|--:|--:|--:|--:|
| AIDO.RNA-650M | 0.2163 | 0.1993 | 0.1767 | 0.2017 |
| AIDO.RNA-1.6B | 0.2070 | 0.2010 | 0.1824 | 0.1989 |
| RNAGenesis | 0.1825 | 0.1802 | 0.1497 | 0.1760 |
| RiNALMo | 0.1690 | 0.1643 | 0.1391 | 0.1611 |
| AIDO.RNA-300M-MARS | 0.1556 | 0.1514 | 0.1369 | 0.1539 |
| AIDO.RNA-25M-MARS | 0.1540 | 0.1380 | 0.1026 | 0.1393 |
| RNA-FM | 0.1293 | 0.1155 | 0.1014 | 0.1267 |
| AIDO.RNA-1M-MARS | 0.0806 | 0.0904 | 0.0800 | 0.0855 |
| Orthrus | 0.0717 | 0.0589 | 0.0384 | 0.0356 |

`wt-fill` has the highest observed macro on 8 of the 9 checkpoints, and it is the convention the
leaderboard uses. The table above is published so the choice can be checked rather than taken on
trust.

Two things to keep in mind when reading it:

- **The differences are concentrated in tRNA.** Mean spread across the four strategies is 0.094 in
  tRNA against 0.021 in ribozyme and 0.007 in aptamer, because the conventions are identical on
  single mutants and the tRNA assays are the deepest (4.07 mutations per variant against 2.91 and
  1.68). Since the macro weights 3 tRNA assays as heavily as 26 ribozyme assays, that effect is
  amplified: under an unweighted mean over all 31 assays the ordering is much less clear cut.
- **Model ranking is far more stable than the absolute numbers.** Across the nine checkpoints the
  only ordering change between strategies is that AIDO.RNA-650M and AIDO.RNA-1.6B trade places.

## Why we now exclude the mRNA assays

We restrict evaluation to assays whose DMS readout is a direct measure of the RNA molecule's own
function (ribozyme cleavage, tRNA function, aptamer binding).

1. mRNA-coding (37 assays): the readout is protein function rather than mRNA fitness.
2. mRNA-splicing (2 assays, `Julien_2016_mRNA` and `Ke_2017_mRNA`): the readout is exon or alternative splicing efficiency in human cells, a regulatory phenotype rather than the molecular fitness of the mRNA itself.

Signed Spearman on the excluded assays is kept for reference in
[`mRNA/leaderboard_signed_mRNA.csv`][4]. Those numbers predate the
RiNALMo and RNA-FM rescoring, so their rows there still use the earlier scoring, and RiNALMo's
position in that table in particular should not be read as current.

## Notes

Categories: ribozyme (26 assays), tRNA (3), aptamer (2). Because the macro-mean weights each category
equally, the small tRNA and aptamer sets carry outsized, higher-variance weight. Treat differences of
a few hundredths as unresolved rather than as an ordering.

## Scoring scripts

Scoring scripts, paths relative to the repository root:
- Evo 2 40B: `fitness/baselines/Evo/score_evo2_single_dms.py` and `score_evo2.sh`
- Orthrus: `fitness/baselines/Orthrus/score_orthrus_single_dms.py` and `score_orthrus.sh`
- AIDO.RNA: `fitness/baselines/AIDO_RNA/score_aido_rna_single_dms.py` and `score_aido_rna.sh`
- RNAGenesis: `fitness/baselines/RNAGenesis/score_rnagenesis_single_dms.py` and `score_rnagenesis.sh`
- RNA-FM: `fitness/baselines/RNA_FM/score_rna_fm_single_dms.py` and `score_rna_fm.sh`
- RiNALMo: `fitness/baselines/RiNALMo/score_rinalmo_single_dms.py` and `score_rinalmo.sh`

The five masked model adapters share one scoring engine, `fitness/baselines/masked_lm`, which implements the
four fill strategies, the context deduplication and the batching. Each model's script is a thin
adapter supplying its alphabet, tokenization and forward pass. `tests/test_fitness.py` covers the
checkpoint-free scoring, merge and analysis workflow on fixtures from real assays.

All are registered in `fitness/merge_scoring_files.py` and `fitness/performance_fitness.py`.

Reproduce the aggregate with `performance_fitness.py --type ncRNA`, whose Spearman is now signed.
It previously reported the absolute value, which credited a model whose scores anti-correlate with
fitness exactly as much as one that correlates, so it could not produce the numbers this page
publishes. Its per-category means now match the columns above, and the macro is their unweighted
mean. AUC and MCC are directed too: an AUC below 0.5 or an MCC below 0 means the model ranks variants
the wrong way round, where both were previously folded onto their better side.

`fitness/analyze_fill_strategies.py` regenerates the sensitivity table and category spreads from the
prediction files.

Scores computed in bfloat16 depend on the GPU: the same code and checkpoint on an L40S and an H100
differ by up to 0.2 in score and about 0.001 in per-assay Spearman. Every prediction file is written
with a manifest recording the strategies, alphabet, model arguments and hardware. All numbers on
this page were produced on H100s.

[0]: leaderboard_signed_3ncRNA.csv
[1]: https://huggingface.co/antichronology/orthrus-mlm-6-track/blob/5f0dc87d51065035fc28e71972c69f9c84f4deae/orthrus_hf.py#L290-L325
[2]: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L186-L225
[3]: https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/esm/compute_fitness.py#L486-L514
[4]: mRNA/leaderboard_signed_mRNA.csv
