# Fitness leaderboard

<!-- BEGIN GENERATED TABLES -->
| Rank | Model | Ribozyme (n=24) | tRNA (n=3) | Aptamer (n=2) | Macro (3 ncRNA) | Depth macro |
|---:|:--|--:|--:|--:|--:|--:|
| 1 | _RNA-2Di_ | 0.53 | 0.63 | 0.26 | 0.47 | 0.08 |
| 2 | _Mutation count_ | 0.47 | 0.63 | 0.12 | 0.41 | 0.00 |
| 3 | _EVmutation_\* | 0.38 | 0.48 | 0.12 | 0.33 | 0.15 |
| 4 | RNA-ERNIE | 0.26 | 0.57 | -0.14 | 0.23 | 0.01 |
| 5 | AIDO.RNA (1.6B) | 0.22 | 0.63 | -0.29 | 0.19 | 0.07 |
| 6 | Evo 2 (7B) | 0.15 | 0.58 | -0.24 | 0.16 | -0.05 |
| 7 | AIDO.RNA (650M) | 0.25 | 0.65 | -0.44 | 0.15 | -0.02 |
| 8 | RiNALMo | 0.05 | 0.64 | -0.27 | 0.14 | 0.10 |
| 9 | RNAGenesis | 0.16 | 0.66 | -0.43 | 0.13 | 0.01 |
| 10 | Evo 2 (20B) | 0.17 | 0.61 | -0.40 | 0.13 | 0.03 |
| 11 | Evo 2 (40B) | 0.14 | 0.61 | -0.40 | 0.12 | 0.00 |
| 12 | Evo 2 (1B base) | 0.04 | 0.57 | -0.33 | 0.10 | 0.02 |
| 13 | RNA-FM | 0.10 | 0.56 | -0.37 | 0.09 | 0.02 |
| 14 | Evo 1.5 | 0.06 | 0.57 | -0.41 | 0.07 | 0.06 |
| 15 | AIDO.RNA (300M) | -0.02 | 0.52 | -0.49 | 0.00 | 0.03 |
| 16 | AIDO.RNA (25M) | -0.07 | 0.53 | -0.50 | -0.01 | -0.01 |
| 17 | Nucleotide Transformer v3 (8M) | 0.03 | 0.39 | -0.50 | -0.03 | 0.02 |
| 18 | Nucleotide Transformer v3 (650M) | -0.01 | 0.45 | -0.55 | -0.04 | 0.04 |
| 19 | Nucleotide Transformer v3 (100M) | 0.00 | 0.43 | -0.55 | -0.04 | 0.03 |
| 20 | GenSLM (2.5B) | -0.05 | 0.04 | -0.12 | -0.04 | -0.02 |
| 21 | Evo 1 | -0.04 | 0.06 | -0.33 | -0.10 | 0.05 |
| 22 | AIDO.RNA (1M) | 0.02 | 0.15 | -0.51 | -0.11 | -0.05 |
| 23 | Orthrus | -0.05 | -0.11 | -0.24 | -0.14 | -0.14 |

Assays of the same RNA molecule are averaged before their category (Domingo_2018_tRNA with Li_2016_tRNA, Zhang_2024_line1_mini_ribozyme with Zhang_2024_line1_full_ribozyme), as ProteinGym averages the assays of one protein.

\* EVmutation scores 27/29 assays and 387/498 variants (77.7%). Its row uses those variants. [Coverage by assay][5]. [Compare all models on those variants][6].
<!-- END GENERATED TABLES -->

[5]: evmutation_coverage.csv
[6]: leaderboard_evmutation.csv
