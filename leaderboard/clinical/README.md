# Clinical leaderboard

Zero-shot AUROC for telling pathogenic from benign human variants, from model
likelihoods alone. ClinVar variants with two or more review stars are read on
the pre-mRNA (10,101 nt of the gene's strand) and on the spliced mRNA, SNVs and
indels apart. AUROC is computed within each consequence class with at least 10
pathogenic and 10 benign variants, so telling classes apart, such as nonsense
from synonymous, earns nothing. A unit's macro averages its classes, and the
ClinVar score averages the four units. Mitochondrial variants from MITOMAP are
read on the processed RNA, and the mito score averages the tRNA and mRNA
missense AUROC. Every model is compared on the variants all models scored.
Higher is better; 0.5 is chance. [How the scores are made][3].

<!-- BEGIN GENERATED TABLES -->
### ClinVar

Macro AUROC within consequence classes, then the mean of the four units. [95% intervals][1], [every class][2].

| Rank | Model | ClinVar | Pre-mRNA SNV (11 classes, 336,535 variants) | Mature mRNA SNV (8 classes, 245,431 variants) | Pre-mRNA indel (7 classes, 40,956 variants) | Mature mRNA indel (4 classes, 33,987 variants) |
|---:|:--|--:|--:|--:|--:|--:|
| 1 | Evo 2 (7B) | 0.78 | 0.82 | 0.70 | 0.82 | 0.76 |
| 2 | Evo 2 (40B) | 0.77 | 0.81 | 0.70 | 0.83 | 0.72 |
| 3 | Evo 2 (20B) | 0.77 | 0.81 | 0.70 | 0.83 | 0.72 |
| 4 | Evo 2 (1B base) | 0.76 | 0.80 | 0.69 | 0.79 | 0.75 |
| 5 | Nucleotide Transformer v3 (650M) | 0.66 | 0.70 | 0.64 | 0.67 | 0.61 |
| 6 | AIDO.RNA (300M) | 0.59 | 0.58 | 0.62 | 0.59 | 0.56 |
| 7 | Nucleotide Transformer v3 (100M) | 0.59 | 0.60 | 0.57 | 0.64 | 0.54 |
| 8 | Nucleotide Transformer v3 (8M) | 0.58 | 0.56 | 0.55 | 0.63 | 0.57 |
| 9 | RiNALMo | 0.57 | 0.56 | 0.64 | 0.54 | 0.52 |
| 10 | AIDO.RNA (25M) | 0.53 | 0.50 | 0.58 | 0.51 | 0.54 |
| 11 | AIDO.RNA (1.6B) | 0.53 | 0.51 | 0.60 | 0.49 | 0.50 |
| 12 | AIDO.RNA (1M) | 0.52 | 0.47 | 0.53 | 0.52 | 0.55 |
| 13 | AIDO.RNA (650M) | 0.52 | 0.50 | 0.59 | 0.49 | 0.49 |
| 14 | RNAGenesis | 0.51 | 0.50 | 0.58 | 0.48 | 0.47 |
| 15 | Orthrus | 0.50 | 0.48 | 0.52 | 0.50 | 0.52 |
| 16 | RNA-FM | 0.49 | 0.48 | 0.58 | 0.47 | 0.44 |

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

Notes:

* Evo 2 scores are the change in mean log-likelihood per token over the window.
  In the mature mRNA and mitochondrial RNA views the window is the whole
  transcript when it fits, so the score depends on its length. How to reduce
  Evo 2 scores in these views is not settled.
* The tRNA class is read on the tRNA with its CCA tail, which the mature tRNA
  carries and mitochondrial DNA does not encode.
* ClinVar substitutions of two or more bases that keep the length are not in
  this release.
* The ClinVar and mito scores have no interval.
* The evaluation tables also hold pooled AUROCs and a consequence-only baseline
  (the out-of-fold pathogenic rate of each variant's consequence), which are not
  on the leaderboard.
* The published scores were made before this code was packaged. Rescored with
  it in the fitness environments, 1,261 variants (775 from ClinVar and all 486
  mitochondrial ones) get exactly the scores of the same code in the
  environments that made the published scores. Against the published scores,
  every Evo 2 and every mitochondrial score is identical, and so are the mito
  AUROCs; masked models' ClinVar scores differ through batch composition, by at
  most 0.094 (RNAGenesis, in bf16), with Spearman at least 0.99996. [By model
  and file][4], [mito AUROCs][5].

[1]: units.csv
[2]: classes.csv
[3]: ../../rnagym/clinical/README.md
[4]: rescoring.csv
[5]: rescoring_mito.csv
