# Clinical leaderboard

<!-- BEGIN GENERATED TABLES -->
### ClinVar

Macro AUROC within consequence classes, then the mean of the four units. [95% intervals][1], [every class][2].

| Rank | Model | ClinVar | Pre-mRNA SNV (3 classes, 104 variants) | Mature mRNA SNV (3 classes, 84 variants) | Pre-mRNA indel (3 classes, 84 variants) | Mature mRNA indel (3 classes, 84 variants) |
|---:|:--|--:|--:|--:|--:|--:|
| 1 | Evo 2 (7B) | 0.80 | 0.97 | 0.65 | 0.79 | 0.78 |
| 2 | Evo 2 (1B base) | 0.79 | 0.94 | 0.68 | 0.79 | 0.74 |
| 3 | Evo 2 (40B) | 0.79 | 0.93 | 0.65 | 0.77 | 0.79 |
| 4 | Evo 2 (20B) | 0.78 | 0.93 | 0.64 | 0.76 | 0.79 |
| 5 | Nucleotide Transformer v3 (650M) | 0.67 | 0.79 | 0.64 | 0.64 | 0.59 |
| 6 | Nucleotide Transformer v3 (100M) | 0.57 | 0.68 | 0.56 | 0.59 | 0.47 |
| 7 | Nucleotide Transformer v3 (8M) | 0.56 | 0.61 | 0.52 | 0.56 | 0.53 |
| 8 | AIDO.RNA (300M) | 0.52 | 0.58 | 0.58 | 0.48 | 0.45 |
| 9 | Orthrus | 0.48 | 0.45 | 0.53 | 0.47 | 0.45 |
| 10 | AIDO.RNA (1M) | 0.47 | 0.44 | 0.46 | 0.49 | 0.48 |
| 11 | RiNALMo | 0.46 | 0.53 | 0.53 | 0.42 | 0.37 |
| 12 | AIDO.RNA (25M) | 0.45 | 0.50 | 0.46 | 0.44 | 0.41 |
| 13 | RNAGenesis | 0.45 | 0.46 | 0.50 | 0.44 | 0.42 |
| 14 | RNA-FM | 0.43 | 0.46 | 0.43 | 0.47 | 0.37 |
| 15 | AIDO.RNA (1.6B) | 0.43 | 0.46 | 0.45 | 0.44 | 0.35 |
| 16 | AIDO.RNA (650M) | 0.42 | 0.46 | 0.46 | 0.41 | 0.35 |

### Mitochondrial (MITOMAP)

Mean of the tRNA and mRNA missense AUROC, on the processed RNA. Strata with fewer than 10 pathogenic or 10 benign variants are left out of the mito score: loss of function (7 P, 1 B), rRNA (2 P, 36 B), synonymous (1 P, 178 B), other coding (0 P, 3 B).

| Rank | Model | Mito | tRNA + CCA (44 P, 18 B) | mRNA missense (41 P, 74 B) |
|---:|:--|--:|--:|--:|
| 1 | Evo 2 (40B) | 0.83 | 0.78 | 0.88 |
| 2 | Evo 2 (20B) | 0.83 | 0.78 | 0.87 |
| 3 | Evo 2 (7B) | 0.83 | 0.71 | 0.95 |
| 4 | Evo 2 (1B base) | 0.82 | 0.73 | 0.91 |
| 5 | AIDO.RNA (300M) | 0.73 | 0.66 | 0.80 |
| 6 | AIDO.RNA (1.6B) | 0.73 | 0.91 | 0.54 |
| 7 | AIDO.RNA (650M) | 0.71 | 0.92 | 0.50 |
| 8 | RiNALMo | 0.66 | 0.85 | 0.47 |
| 9 | Nucleotide Transformer v3 (650M) | 0.61 | 0.40 | 0.83 |
| 10 | RNA-FM | 0.59 | 0.80 | 0.38 |
| 11 | RNAGenesis | 0.59 | 0.74 | 0.43 |
| 12 | AIDO.RNA (25M) | 0.52 | 0.52 | 0.52 |
| 13 | Nucleotide Transformer v3 (100M) | 0.47 | 0.48 | 0.47 |
| 14 | Orthrus | 0.47 | 0.56 | 0.39 |
| 15 | Nucleotide Transformer v3 (8M) | 0.46 | 0.48 | 0.44 |
| 16 | AIDO.RNA (1M) | 0.45 | 0.42 | 0.48 |
<!-- END GENERATED TABLES -->

[1]: units.csv
[2]: classes.csv
