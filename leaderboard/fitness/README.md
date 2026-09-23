# Fitness leaderboard

Signed Spearman correlation with experimental fitness on 29 ncRNA assays.
Columns average assays within each category. Macro weights the three categories
equally. Higher is better.

<!-- BEGIN GENERATED TABLES -->
| Rank | Model | Ribozyme (n=24) | tRNA (n=3) | Aptamer (n=2) | Macro (3 ncRNA) |
|---:|:--|--:|--:|--:|--:|
| 1 | EVmutation* | 0.2460 | 0.4602 | 0.1371 | 0.2811 |
| 2 | AIDO.RNA (650M) | 0.0577 | 0.4894 | 0.0934 | 0.2135 |
| 3 | Evo 2 (20B) | 0.1041 | 0.4345 | 0.0937 | 0.2108 |
| 4 | Evo 2 (40B) | 0.1017 | 0.4310 | 0.0970 | 0.2099 |
| 5 | AIDO.RNA (1.6B) | 0.0453 | 0.4884 | 0.0718 | 0.2018 |
| 6 | Evo 2 (1B base) | 0.0028 | 0.4466 | 0.1334 | 0.1943 |
| 7 | RNA-ERNIE | 0.1353 | 0.4161 | 0.0306 | 0.1940 |
| 8 | Evo 2 (7B) | 0.0617 | 0.3867 | 0.1192 | 0.1892 |
| 9 | RNAGenesis | 0.0656 | 0.4381 | 0.0343 | 0.1793 |
| 10 | RiNALMo | -0.0148 | 0.4856 | 0.0459 | 0.1722 |
| 11 | AIDO.RNA (25M) | -0.0190 | 0.4569 | 0.0348 | 0.1576 |
| 12 | AIDO.RNA (300M) | -0.0224 | 0.4551 | 0.0372 | 0.1566 |
| 13 | Evo 1.5 | 0.0495 | 0.3849 | -0.0010 | 0.1445 |
| 14 | RNA-FM | -0.0090 | 0.4147 | -0.0043 | 0.1338 |
| 15 | Nucleotide Transformer v3 (8M) | -0.0075 | 0.3053 | 0.0988 | 0.1322 |
| 16 | Nucleotide Transformer v3 (650M) | -0.0177 | 0.3518 | 0.0436 | 0.1259 |
| 17 | Nucleotide Transformer v3 (100M) | -0.0118 | 0.2544 | 0.1001 | 0.1143 |
| 18 | AIDO.RNA (1M) | 0.0109 | 0.1777 | 0.0626 | 0.0837 |
| 19 | Orthrus | -0.0414 | 0.0922 | 0.1578 | 0.0695 |
| 20 | GenSLM (2.5B) | -0.0214 | 0.1002 | 0.0620 | 0.0469 |
| 21 | Evo 1 | -0.0129 | 0.0989 | 0.0300 | 0.0386 |

\* EVmutation scores 27/29 assays and 485,150/825,011 variants (58.8%). Its row uses those variants. [Coverage by assay][5]. [Compare all models on those variants][6].
<!-- END GENERATED TABLES -->

[Full precision CSV][0]. Small differences should be read cautiously, especially
with only three tRNA and two aptamer assays. Coding and splicing assays are
excluded because they measure protein function or splicing efficiency.

## Scoring

Masked language models use `wt-fill`: mask each mutated position in the wild-type
sequence and sum `log p(mutant) - log p(wild type)`. This follows the [ESM][1]
and [ProteinGym][2] implementations. All four [fill strategies][3] are available
in the prediction files. RNA-ERNIE uses unmasked wild-type probability changes.
GenSLM uses mean next-codon log likelihood, excluding padding.

## Reproduce

Follow the [fitness setup][4], then run from `rnagym/fitness/`:

```bash
pixi run merge
pixi run leaderboard
pixi run --locked -e default check-published
```

`pixi run analyze-fill-strategies` compares the four scoring conventions.

[0]: leaderboard_signed_3ncRNA.csv
[1]: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L186-L225
[2]: https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/esm/compute_fitness.py#L486-L514
[3]: ../../rnagym/fitness/baselines/masked_lm/README.md
[4]: ../../rnagym/fitness/README.md
[5]: evmutation_coverage.csv
[6]: leaderboard_evmutation.csv
