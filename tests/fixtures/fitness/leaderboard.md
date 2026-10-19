# Fitness leaderboard

<!-- BEGIN GENERATED TABLES -->
| Rank | Model | Ribozyme (24 assays, 23 molecules) | tRNA (3 assays, 2 molecules) | Aptamer (2 assays, 2 molecules) | Macro (3 ncRNA) |
|---:|:--|--:|--:|--:|--:|
| 1 | _RNA-2Di_ | 0.53 | 0.63 | 0.26 | 0.47 |
| 2 | _Mutation count_ | 0.47 | 0.63 | 0.12 | 0.41 |
| 3 | _EVmutation_\* | 0.38 | 0.48 | 0.12 | 0.33 |
| 4 | RNA-ERNIE | 0.26 | 0.57 | -0.14 | 0.23 |
| 5 | AIDO.RNA (1.6B) | 0.22 | 0.63 | -0.29 | 0.19 |
| 6 | Evo 2 (7B) | 0.15 | 0.58 | -0.24 | 0.16 |
| 7 | AIDO.RNA (650M) | 0.25 | 0.65 | -0.44 | 0.15 |
| 8 | RiNALMo | 0.05 | 0.64 | -0.27 | 0.14 |
| 9 | RNAGenesis | 0.16 | 0.66 | -0.43 | 0.13 |
| 10 | Evo 2 (20B) | 0.17 | 0.61 | -0.40 | 0.13 |
| 11 | Evo 2 (40B) | 0.14 | 0.61 | -0.40 | 0.12 |
| 12 | Evo 2 (1B base) | 0.04 | 0.57 | -0.33 | 0.10 |
| 13 | RNA-FM | 0.10 | 0.56 | -0.37 | 0.09 |
| 14 | Evo 1.5 | 0.06 | 0.57 | -0.41 | 0.07 |
| 15 | AIDO.RNA (300M) | -0.02 | 0.52 | -0.49 | 0.00 |
| 16 | AIDO.RNA (25M) | -0.07 | 0.53 | -0.50 | -0.01 |
| 17 | Nucleotide Transformer v3 (8M) | 0.03 | 0.39 | -0.50 | -0.03 |
| 18 | Nucleotide Transformer v3 (650M) | -0.01 | 0.45 | -0.55 | -0.04 |
| 19 | Nucleotide Transformer v3 (100M) | 0.00 | 0.43 | -0.55 | -0.04 |
| 20 | GenSLM (2.5B) | -0.05 | 0.04 | -0.12 | -0.04 |
| 21 | Evo 1 | -0.04 | 0.06 | -0.33 | -0.10 |
| 22 | AIDO.RNA (1M) | 0.02 | 0.15 | -0.51 | -0.11 |
| 23 | Orthrus | -0.05 | -0.11 | -0.24 | -0.14 |

Assays of the same RNA molecule are averaged before their category (Domingo_2018_tRNA with Li_2016_tRNA, Zhang_2024_line1_mini_ribozyme with Zhang_2024_line1_full_ribozyme), as ProteinGym averages the assays of one protein.

\* EVmutation scores 27/29 assays and 387/498 variants (77.7%). Its row uses those variants. [Coverage by assay][5]. [Compare all models on those variants][6].

### By mutation count

Spearman among variants with the same number of mutations within each assay, for counts with at least 30 variants and varying measured fitness, so the number of mutations alone ranks nothing. Assays of one molecule are averaged first, then molecules within each category. Counts are never averaged together. Molecule counts are those of the models scored on every variant. EVmutation\* uses only the variants it scores, so its cells can rest on fewer molecules. Mutation count is left out, as it is constant within a count. [All counts][8].

Ribozyme, by mutation count (molecules):

| Model | 1 (2) | 2 (1) |
|:--|--:|--:|
| _RNA-2Di_ | 0.06 | 0.13 |
| _EVmutation_\* | 0.30 | 0.14 |
| RNA-ERNIE | -0.03 | 0.10 |
| AIDO.RNA (1.6B) | 0.17 | 0.36 |
| Evo 2 (7B) | 0.00 | 0.28 |
| AIDO.RNA (650M) | 0.11 | 0.29 |
| RiNALMo | 0.25 | 0.28 |
| RNAGenesis | 0.22 | -0.06 |
| Evo 2 (20B) | 0.05 | 0.27 |
| Evo 2 (40B) | -0.03 | 0.27 |
| Evo 2 (1B base) | 0.08 | 0.40 |
| RNA-FM | 0.13 | 0.08 |
| Evo 1.5 | 0.36 | 0.50 |
| AIDO.RNA (300M) | 0.12 | 0.47 |
| AIDO.RNA (25M) | 0.10 | 0.48 |
| Nucleotide Transformer v3 (8M) | 0.30 | 0.26 |
| Nucleotide Transformer v3 (650M) | 0.20 | 0.36 |
| Nucleotide Transformer v3 (100M) | 0.21 | 0.48 |
| GenSLM (2.5B) | -0.02 | 0.10 |
| Evo 1 | 0.33 | 0.58 |
| AIDO.RNA (1M) | 0.10 | 0.54 |
| Orthrus | -0.18 | -0.05 |

tRNA, by mutation count (molecules):

| Model | 2 (1) | 3 (1) |
|:--|--:|--:|
| _RNA-2Di_ | 0.11 | 0.10 |
| _EVmutation_\* | 0.42 | -0.07 |
| RNA-ERNIE | 0.19 | -0.12 |
| AIDO.RNA (1.6B) | 0.34 | 0.20 |
| Evo 2 (7B) | 0.29 | -0.07 |
| AIDO.RNA (650M) | 0.19 | -0.01 |
| RiNALMo | 0.49 | -0.10 |
| RNAGenesis | 0.14 | 0.24 |
| Evo 2 (20B) | 0.21 | 0.10 |
| Evo 2 (40B) | 0.20 | 0.07 |
| Evo 2 (1B base) | 0.38 | 0.15 |
| RNA-FM | 0.00 | 0.24 |
| Evo 1.5 | 0.12 | 0.11 |
| AIDO.RNA (300M) | 0.23 | 0.11 |
| AIDO.RNA (25M) | 0.18 | 0.07 |
| Nucleotide Transformer v3 (8M) | 0.03 | 0.10 |
| Nucleotide Transformer v3 (650M) | 0.14 | 0.21 |
| Nucleotide Transformer v3 (100M) | 0.24 | 0.03 |
| GenSLM (2.5B) | 0.00 | -0.23 |
| Evo 1 | -0.06 | -0.13 |
| AIDO.RNA (1M) | 0.14 | -0.12 |
| Orthrus | 0.02 | -0.17 |

Aptamer, by mutation count (molecules):

| Model | 1 (1) | 2 (1) |
|:--|--:|--:|
| _RNA-2Di_ | 0.12 | -0.06 |
| _EVmutation_\* | 0.29 | -0.36 |
| RNA-ERNIE | 0.06 | -0.06 |
| AIDO.RNA (1.6B) | -0.12 | -0.33 |
| Evo 2 (7B) | -0.32 | -0.25 |
| AIDO.RNA (650M) | -0.28 | -0.20 |
| RiNALMo | -0.18 | -0.17 |
| RNAGenesis | -0.11 | -0.48 |
| Evo 2 (20B) | 0.03 | -0.26 |
| Evo 2 (40B) | 0.03 | -0.26 |
| Evo 2 (1B base) | -0.22 | -0.31 |
| RNA-FM | -0.22 | -0.09 |
| Evo 1.5 | -0.19 | -0.40 |
| AIDO.RNA (300M) | 0.03 | -0.43 |
| AIDO.RNA (25M) | -0.06 | -0.41 |
| Nucleotide Transformer v3 (8M) | -0.17 | -0.31 |
| Nucleotide Transformer v3 (650M) | -0.10 | -0.35 |
| Nucleotide Transformer v3 (100M) | -0.15 | -0.36 |
| GenSLM (2.5B) | 0.10 | 0.13 |
| Evo 1 | -0.14 | -0.07 |
| AIDO.RNA (1M) | -0.13 | -0.39 |
| Orthrus | -0.25 | -0.09 |
<!-- END GENERATED TABLES -->

[5]: evmutation_coverage.csv
[6]: leaderboard_evmutation.csv
[8]: leaderboard_by_mutation_count.csv
