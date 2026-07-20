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
| 1 | Evo 2 (40B) | 0.1081 | 0.4310 | 0.0970 | 0.2120 |
| 2 | RNA-ERNIE | 0.1343 | 0.4161 | 0.0306 | 0.1937 |
| 3 | Evo 2 (7B) | 0.0651 | 0.3867 | 0.1192 | 0.1904 |
| 4 | RNA-FM | 0.1078 | 0.4638 | -0.1177 | 0.1513 |
| 5 | Evo 1.5 | 0.0278 | 0.3850 | 0.0007 | 0.1378 |
| 6 | Nucleotide Transformer | 0.1329 | 0.3166 | -0.0886 | 0.1203 |
| 7 | RiNALMo | 0.0246 | 0.2597 | 0.0259 | 0.1034 |
| 8 | Orthrus | -0.0276 | -0.0068 | 0.1495 | 0.0384 |
| 9 | Evo 1 | -0.0216 | 0.0948 | 0.0058 | 0.0263 |
| 10 | GenSLM | -0.0045 | -0.0934 | -0.0036 | -0.0338 |

Per category columns are the mean signed Spearman over the assays in that category; Macro is the
unweighted average of the three category means. Underlying values: [`leaderboard_signed_3ncRNA.csv`](leaderboard_signed_3ncRNA.csv).

## Why we now exclude the mRNA assays

We restrict evaluation to assays whose DMS readout is a direct measure of the RNA molecule's own
function (ribozyme cleavage, tRNA function, aptamer binding).

1. mRNA-coding (37 assays): the readout is protein function rather than mRNA fitness.
2. mRNA-splicing (2 assays, `Julien_2016_mRNA` and `Ke_2017_mRNA`): the readout is exon or alternative splicing efficiency in human cells, a regulatory phenotype rather than the molecular fitness of the mRNA itself.

## Notes

Categories: ribozyme (26 assays), tRNA (3), aptamer (2). Because the macro-mean weights each category
equally, the small tRNA and aptamer sets carry outsized, higher-variance weight.

## Scoring scripts

Scoring scripts for the two new baselines are on the [`v0.2`](https://github.com/MarksLab-DasLab/RNAGym/tree/v0.2) branch:
- Evo 2 40B: `fitness/baselines/Evo/score_evo2_single_dms.py` and `score_evo2.sh`
- Orthrus: `fitness/baselines/Orthrus/score_orthrus_single_dms.py` and `score_orthrus.sh`

Both are registered in `fitness/merge_scoring_files.py` and `fitness/performance_fitness.py` on `v0.2`.
