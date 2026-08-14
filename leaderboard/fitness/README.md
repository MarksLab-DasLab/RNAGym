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
| 4 | AIDO.RNA (1.6B) | 0.0740 | 0.3885 | 0.0847 | 0.1824 |
| 5 | RNAGenesis | 0.0465 | 0.3735 | 0.0292 | 0.1498 |
| 6 | RiNALMo | -0.0092 | 0.3910 | 0.0356 | 0.1391 |
| 7 | Evo 1.5 | 0.0278 | 0.3850 | 0.0007 | 0.1378 |
| 8 | Nucleotide Transformer | 0.1329 | 0.3166 | -0.0886 | 0.1203 |
| 9 | RNA-FM | -0.0303 | 0.3342 | 0.0004 | 0.1014 |
| 10 | Orthrus | -0.0276 | -0.0068 | 0.1495 | 0.0384 |
| 11 | Evo 1 | -0.0216 | 0.0948 | 0.0058 | 0.0263 |
| 12 | GenSLM | -0.0045 | -0.0934 | -0.0036 | -0.0338 |

Per category columns are the mean signed Spearman over the assays in that category; Macro is the
unweighted average of the three category means. Underlying values: [`leaderboard_signed_3ncRNA.csv`](leaderboard_signed_3ncRNA.csv).

AIDO.RNA and RNAGenesis are new entries. RiNALMo and RNA-FM were rescored, see below.

## Scoring convention

All masked language models are now scored the same way: the masked-marginal log-likelihood ratio,
with each mutated position masked one at a time in the variant's own sequence, summed over the
variant's mutated positions. This matters because 99.4% of the ncRNA variants are multi-mutants,
which is exactly where scoring conventions differ.

RiNALMo and RNA-FM were rescored to reach that convention, and their earlier scoring scripts were
replaced. The earlier RiNALMo scores looked the mutant base up in an RNA-form sequence, and its
alphabet is DNA, so mutations to U scored against the unknown-token logit. The earlier RNA-FM scores
match a single unmasked wild-type pass rather than the masked strategy its run script requested.

## AIDO.RNA across model sizes

All five released AIDO.RNA checkpoints on the same 31 assays. Values:
[`aido_rna_scaling.csv`](aido_rna_scaling.csv).

| Checkpoint | Params | Ribozyme | tRNA | Aptamer | Macro |
|:--|--:|--:|--:|--:|--:|
| AIDO.RNA-1M-MARS | 1M | 0.0015 | 0.1763 | 0.0624 | 0.0801 |
| AIDO.RNA-25M-MARS | 25M | -0.0009 | 0.2718 | 0.0367 | 0.1025 |
| AIDO.RNA-300M-MARS | 299M | 0.0058 | 0.3684 | 0.0365 | 0.1369 |
| AIDO.RNA-650M | 646M | 0.0460 | 0.3928 | 0.0939 | 0.1775 |
| AIDO.RNA-1.6B | 1606M | 0.0740 | 0.3885 | 0.0847 | 0.1824 |

Performance increases with size at every step. The macro gain from 650M to 1.6B is small (+0.0049),
but that is a property of the weighting rather than of the model: under an equal-assay average over
all 31 assays the same step is +0.0225, the second largest. tRNA and aptamer hold two thirds of the
macro weight and both stop improving at 650M, while ribozyme holds one third and keeps improving.

## Why we now exclude the mRNA assays

We restrict evaluation to assays whose DMS readout is a direct measure of the RNA molecule's own
function (ribozyme cleavage, tRNA function, aptamer binding).

1. mRNA-coding (37 assays): the readout is protein function rather than mRNA fitness.
2. mRNA-splicing (2 assays, `Julien_2016_mRNA` and `Ke_2017_mRNA`): the readout is exon or alternative splicing efficiency in human cells, a regulatory phenotype rather than the molecular fitness of the mRNA itself.

Signed Spearman on the excluded assays is kept for reference in
[`mRNA/leaderboard_signed_mRNA.csv`](mRNA/leaderboard_signed_mRNA.csv). Those numbers predate the
RiNALMo and RNA-FM rescoring, so their rows there still use the earlier scoring, and RiNALMo's
position in that table in particular should not be read as current.

## Notes

Categories: ribozyme (26 assays), tRNA (3), aptamer (2). Because the macro-mean weights each category
equally, the small tRNA and aptamer sets carry outsized, higher-variance weight. Ranks that differ by
less than about 0.01 should not be read as meaningful separations.

## Scoring scripts

Scoring scripts are on the [`v0.2`](https://github.com/MarksLab-DasLab/RNAGym/tree/v0.2) branch:
- Evo 2 40B: `fitness/baselines/Evo/score_evo2_single_dms.py` and `score_evo2.sh`
- Orthrus: `fitness/baselines/Orthrus/score_orthrus_single_dms.py` and `score_orthrus.sh`
- AIDO.RNA: `fitness/baselines/AIDO_RNA/score_aido_rna_single_dms.py` and `score_aido_rna.sh`
- RNAGenesis: `fitness/baselines/RNAGenesis/score_rnagenesis_single_dms.py` and `score_rnagenesis.sh`
- RNA-FM: `fitness/baselines/RNA_FM/score_rna_fm_single_dms.py` and `score_rna_fm.sh`
- RiNALMo: `fitness/baselines/RiNALMo/score_rinalmo_single_dms.py` and `score_rinalmo.sh`

All are registered in `fitness/merge_scoring_files.py` and `fitness/performance_fitness.py` on `v0.2`.
Reproduce the aggregate with `performance_fitness.py --type ncRNA`.
