# Fitness leaderboard

Signed Spearman correlation with experimental fitness on 29 ncRNA assays of 27
RNA molecules. Assays of the same molecule are averaged first, as ProteinGym
averages the assays of one protein. Columns then average molecules within each
category, and Macro weights the three categories equally. Higher is better.

<!-- BEGIN GENERATED TABLES -->
| Rank | Model | Ribozyme (24 assays, 23 molecules) | tRNA (3 assays, 2 molecules) | Aptamer (2 assays, 2 molecules) | Macro (3 ncRNA) |
|---:|:--|--:|--:|--:|--:|
| 1 | EVmutation* | 0.2474 | 0.4397 | 0.1371 | 0.2747 |
| 2 | AIDO.RNA (650M) | 0.0610 | 0.4663 | 0.0934 | 0.2069 |
| 3 | Evo 2 (20B) | 0.1037 | 0.4206 | 0.0937 | 0.2060 |
| 4 | Evo 2 (40B) | 0.1012 | 0.4177 | 0.0970 | 0.2053 |
| 5 | AIDO.RNA (1.6B) | 0.0475 | 0.4656 | 0.0718 | 0.1950 |
| 6 | Evo 2 (7B) | 0.0636 | 0.3790 | 0.1192 | 0.1873 |
| 7 | Evo 2 (1B base) | 0.0016 | 0.4197 | 0.1334 | 0.1849 |
| 8 | RNA-ERNIE | 0.1363 | 0.3757 | 0.0306 | 0.1809 |
| 9 | RNAGenesis | 0.0684 | 0.4264 | 0.0343 | 0.1764 |
| 10 | RiNALMo | -0.0156 | 0.4654 | 0.0459 | 0.1652 |
| 11 | AIDO.RNA (300M) | -0.0236 | 0.4332 | 0.0372 | 0.1489 |
| 12 | AIDO.RNA (25M) | -0.0199 | 0.4289 | 0.0348 | 0.1479 |
| 13 | Evo 1.5 | 0.0562 | 0.3515 | -0.0010 | 0.1356 |
| 14 | Nucleotide Transformer v3 (8M) | -0.0075 | 0.2713 | 0.0988 | 0.1209 |
| 15 | RNA-FM | -0.0083 | 0.3709 | -0.0043 | 0.1194 |
| 16 | Nucleotide Transformer v3 (650M) | -0.0186 | 0.3254 | 0.0436 | 0.1168 |
| 17 | Nucleotide Transformer v3 (100M) | -0.0128 | 0.2415 | 0.1001 | 0.1096 |
| 18 | AIDO.RNA (1M) | 0.0105 | 0.1383 | 0.0626 | 0.0705 |
| 19 | Orthrus | -0.0427 | 0.0778 | 0.1578 | 0.0643 |
| 20 | Evo 1 | -0.0104 | 0.0882 | 0.0300 | 0.0359 |
| 21 | GenSLM (2.5B) | -0.0205 | 0.0528 | 0.0620 | 0.0315 |

Assays of the same RNA molecule are averaged before their category (Domingo_2018_tRNA with Li_2016_tRNA, Zhang_2024_line1_mini_ribozyme with Zhang_2024_line1_full_ribozyme), as ProteinGym averages the assays of one protein.

\* EVmutation scores 27/29 assays and 485,150/825,011 variants (58.8%). Its row uses those variants. [Coverage by assay][5]. [Compare all models on those variants][6].
<!-- END GENERATED TABLES -->

[Full precision CSV][0]. Small differences should be read cautiously, especially
with only two tRNA and two aptamer molecules. Coding and splicing assays are
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
