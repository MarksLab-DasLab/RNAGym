# Fitness leaderboard

Signed Spearman correlation with experimental fitness on 29 ncRNA assays of 27
RNA molecules. Assays of the same molecule are averaged first, as ProteinGym
averages the assays of one protein. Columns then average molecules within each
category, and Macro weights the three categories equally. The tables by mutation
count rank only variants with the same number of mutations, which shows how well
models rank variants when the naive prior that more mutations are worse (see
Mutation count) no longer helps. Italic rows are simple baselines without
pretraining. Higher is better.

<!-- BEGIN GENERATED TABLES -->
| Rank | Model | Ribozyme (24 assays, 23 molecules) | tRNA (3 assays, 2 molecules) | Aptamer (2 assays, 2 molecules) | Macro (3 ncRNA) |
|---:|:--|--:|--:|--:|--:|
| 1 | _EVmutation_\* | 0.32 | 0.45 | 0.14 | 0.30 |
| 2 | _RNA-2Di_ | 0.25 | 0.47 | -0.07 | 0.22 |
| 3 | AIDO.RNA (650M) | 0.06 | 0.48 | 0.09 | 0.21 |
| 4 | Evo 2 (40B) | 0.10 | 0.42 | 0.10 | 0.21 |
| 5 | Evo 2 (20B) | 0.10 | 0.42 | 0.09 | 0.21 |
| 6 | Evo 2 (7B) | 0.06 | 0.41 | 0.12 | 0.20 |
| 7 | AIDO.RNA (1.6B) | 0.05 | 0.47 | 0.07 | 0.20 |
| 8 | Evo 2 (1B base) | 0.00 | 0.45 | 0.13 | 0.20 |
| 9 | RNA-ERNIE | 0.14 | 0.39 | 0.03 | 0.19 |
| 10 | RNAGenesis | 0.07 | 0.45 | 0.03 | 0.18 |
| 11 | _Mutation count_ | 0.20 | 0.46 | -0.12 | 0.18 |
| 12 | RiNALMo | -0.02 | 0.48 | 0.05 | 0.17 |
| 13 | AIDO.RNA (25M) | -0.02 | 0.46 | 0.03 | 0.16 |
| 14 | RNA-FM | -0.01 | 0.48 | 0.00 | 0.15 |
| 15 | AIDO.RNA (300M) | -0.02 | 0.45 | 0.04 | 0.15 |
| 16 | Evo 1.5 | 0.06 | 0.39 | 0.00 | 0.15 |
| 17 | Nucleotide Transformer v3 (650M) | -0.02 | 0.36 | 0.04 | 0.13 |
| 18 | Nucleotide Transformer v3 (8M) | -0.01 | 0.27 | 0.10 | 0.12 |
| 19 | Nucleotide Transformer v3 (100M) | -0.01 | 0.25 | 0.10 | 0.11 |
| 20 | AIDO.RNA (1M) | 0.01 | 0.24 | 0.06 | 0.11 |
| 21 | Orthrus | -0.04 | 0.08 | 0.16 | 0.06 |
| 22 | Evo 1 | -0.01 | 0.13 | 0.03 | 0.05 |
| 23 | GenSLM (2.5B) | -0.02 | 0.09 | 0.06 | 0.04 |

Assays of the same RNA molecule are averaged before their category (Domingo_2018_tRNA with Li_2016_tRNA, Zhang_2024_line1_mini_ribozyme with Zhang_2024_line1_full_ribozyme), as ProteinGym averages the assays of one protein.

\* EVmutation scores 22/29 assays and 490,522/825,011 variants (59.5%). Its row uses those variants. [Coverage by assay][5]. [Compare all models on those variants][6].

### By mutation count

Spearman among variants with the same number of mutations within each assay, for counts with at least 30 variants and varying measured fitness, so the number of mutations alone ranks nothing. Assays of one molecule are averaged first, then molecules within each category. Counts are never averaged together. Molecule counts are those of the models scored on every variant. EVmutation\* uses only the variants it scores, so its cells can rest on fewer molecules. Mutation count is left out, as it is constant within a count. [All counts][8].

Ribozyme, by mutation count (molecules):

| Model | 1 (17) | 2 (22) | 3 (12) | 4 (12) | 5 (10) | 6 (8) | 7 (8) | 8 (5) | 9 (4) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| _EVmutation_\* | 0.23 | 0.17 | 0.13 | 0.13 | -0.01 | -0.01 | -0.03 | 0.03 | -0.07 |
| _RNA-2Di_ | 0.18 | 0.21 | 0.20 | 0.15 | 0.11 | 0.11 | 0.11 | 0.09 | 0.06 |
| AIDO.RNA (650M) | 0.04 | 0.04 | 0.05 | 0.04 | 0.00 | 0.03 | 0.05 | 0.02 | 0.03 |
| Evo 2 (40B) | 0.08 | 0.07 | 0.08 | 0.09 | 0.04 | 0.06 | 0.05 | 0.03 | 0.05 |
| Evo 2 (20B) | 0.08 | 0.07 | 0.08 | 0.09 | 0.05 | 0.06 | 0.05 | 0.03 | 0.05 |
| Evo 2 (7B) | 0.02 | 0.02 | 0.05 | 0.05 | 0.03 | 0.02 | 0.01 | 0.01 | 0.03 |
| AIDO.RNA (1.6B) | 0.07 | 0.03 | 0.04 | 0.01 | 0.03 | 0.00 | 0.00 | -0.01 | 0.01 |
| Evo 2 (1B base) | -0.04 | -0.03 | -0.05 | -0.04 | -0.01 | -0.02 | -0.03 | 0.01 | 0.04 |
| RNA-ERNIE | 0.02 | 0.02 | -0.02 | 0.01 | 0.05 | 0.06 | 0.10 | 0.01 | 0.02 |
| RNAGenesis | 0.08 | 0.01 | 0.03 | -0.02 | 0.03 | 0.03 | 0.02 | 0.03 | 0.09 |
| RiNALMo | -0.03 | -0.03 | -0.01 | 0.00 | 0.03 | 0.01 | 0.01 | 0.06 | 0.05 |
| AIDO.RNA (25M) | -0.01 | -0.05 | -0.05 | -0.06 | 0.05 | 0.09 | 0.12 | 0.14 | 0.07 |
| RNA-FM | -0.03 | -0.08 | -0.06 | -0.09 | 0.01 | 0.00 | 0.02 | 0.04 | 0.00 |
| AIDO.RNA (300M) | -0.03 | -0.05 | -0.06 | -0.07 | 0.05 | 0.07 | 0.11 | 0.13 | 0.09 |
| Evo 1.5 | -0.02 | -0.02 | 0.04 | 0.03 | 0.02 | 0.06 | 0.08 | 0.05 | 0.06 |
| Nucleotide Transformer v3 (650M) | -0.04 | -0.06 | -0.03 | -0.05 | 0.02 | 0.05 | 0.08 | 0.09 | -0.02 |
| Nucleotide Transformer v3 (8M) | -0.01 | -0.05 | -0.05 | -0.02 | 0.02 | 0.03 | 0.08 | 0.06 | 0.04 |
| Nucleotide Transformer v3 (100M) | -0.01 | -0.06 | -0.08 | -0.07 | 0.01 | 0.02 | 0.04 | 0.07 | 0.06 |
| AIDO.RNA (1M) | 0.01 | -0.01 | -0.06 | -0.07 | 0.02 | 0.04 | 0.08 | 0.11 | 0.05 |
| Orthrus | -0.01 | -0.02 | -0.03 | -0.04 | -0.01 | 0.00 | -0.01 | -0.04 | -0.01 |
| Evo 1 | -0.01 | -0.06 | -0.04 | -0.05 | 0.01 | 0.00 | 0.03 | -0.02 | 0.04 |
| GenSLM (2.5B) | -0.04 | -0.04 | -0.03 | -0.05 | -0.02 | -0.01 | -0.01 | -0.02 | 0.08 |

tRNA, by mutation count (molecules):

| Model | 1 (2) | 2 (2) | 3 (2) | 4 (2) | 5 (2) | 6 (2) | 7 (2) | 8 (2) | 9 (1) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| _EVmutation_\* | 0.44 | 0.39 | 0.20 | 0.09 | 0.08 | 0.06 | 0.07 | 0.05 | 0.08 |
| _RNA-2Di_ | 0.31 | 0.23 | 0.06 | 0.00 | -0.01 | 0.02 | 0.07 | 0.11 | 0.09 |
| AIDO.RNA (650M) | 0.54 | 0.39 | 0.15 | 0.04 | 0.04 | 0.03 | 0.02 | 0.05 | 0.08 |
| Evo 2 (40B) | 0.39 | 0.34 | 0.18 | 0.08 | 0.06 | 0.08 | 0.00 | 0.03 | -0.01 |
| Evo 2 (20B) | 0.39 | 0.34 | 0.18 | 0.08 | 0.06 | 0.08 | 0.00 | 0.02 | -0.01 |
| Evo 2 (7B) | 0.42 | 0.36 | 0.18 | 0.09 | 0.07 | 0.07 | -0.01 | 0.00 | 0.17 |
| AIDO.RNA (1.6B) | 0.54 | 0.37 | 0.14 | 0.03 | 0.02 | 0.03 | 0.02 | 0.05 | 0.08 |
| Evo 2 (1B base) | 0.50 | 0.41 | 0.21 | 0.12 | 0.10 | 0.10 | 0.04 | 0.04 | 0.11 |
| RNA-ERNIE | 0.32 | 0.27 | 0.14 | 0.12 | 0.11 | 0.10 | 0.07 | 0.05 | 0.33 |
| RNAGenesis | 0.45 | 0.31 | 0.11 | 0.00 | 0.00 | -0.02 | -0.05 | 0.04 | 0.07 |
| RiNALMo | 0.64 | 0.45 | 0.18 | 0.07 | 0.05 | 0.06 | 0.00 | 0.03 | 0.02 |
| AIDO.RNA (25M) | 0.53 | 0.39 | 0.17 | 0.08 | 0.08 | 0.08 | 0.06 | 0.08 | 0.17 |
| RNA-FM | 0.59 | 0.43 | 0.15 | 0.05 | 0.02 | 0.04 | 0.03 | 0.09 | 0.08 |
| AIDO.RNA (300M) | 0.54 | 0.36 | 0.14 | 0.05 | 0.05 | 0.07 | 0.05 | 0.04 | 0.12 |
| Evo 1.5 | 0.37 | 0.37 | 0.18 | 0.11 | 0.09 | 0.09 | 0.01 | 0.11 | 0.15 |
| Nucleotide Transformer v3 (650M) | 0.47 | 0.35 | 0.18 | 0.08 | 0.07 | 0.06 | 0.03 | -0.02 | 0.04 |
| Nucleotide Transformer v3 (8M) | 0.30 | 0.22 | 0.12 | 0.07 | 0.08 | 0.06 | 0.01 | 0.07 | 0.15 |
| Nucleotide Transformer v3 (100M) | 0.31 | 0.26 | 0.13 | 0.08 | 0.09 | 0.07 | 0.02 | 0.11 | 0.26 |
| AIDO.RNA (1M) | 0.30 | 0.27 | 0.14 | 0.09 | 0.10 | 0.08 | 0.03 | 0.10 | 0.12 |
| Orthrus | 0.11 | 0.08 | 0.05 | 0.04 | 0.02 | 0.01 | 0.07 | 0.08 | 0.25 |
| Evo 1 | 0.20 | 0.07 | -0.01 | -0.05 | -0.04 | -0.02 | -0.01 | -0.01 | 0.10 |
| GenSLM (2.5B) | -0.07 | 0.07 | 0.08 | 0.08 | 0.10 | 0.10 | 0.06 | 0.07 | 0.06 |

Aptamer, by mutation count (molecules):

| Model | 1 (2) | 2 (2) |
|:--|--:|--:|
| _EVmutation_\* | 0.19 | 0.09 |
| _RNA-2Di_ | 0.17 | 0.04 |
| AIDO.RNA (650M) | 0.08 | 0.04 |
| Evo 2 (40B) | 0.11 | 0.04 |
| Evo 2 (20B) | 0.11 | 0.04 |
| Evo 2 (7B) | 0.08 | 0.02 |
| AIDO.RNA (1.6B) | 0.10 | -0.02 |
| Evo 2 (1B base) | 0.07 | 0.06 |
| RNA-ERNIE | 0.21 | 0.04 |
| RNAGenesis | 0.04 | -0.04 |
| RiNALMo | 0.04 | 0.01 |
| AIDO.RNA (25M) | 0.06 | -0.06 |
| RNA-FM | -0.04 | 0.00 |
| AIDO.RNA (300M) | 0.05 | -0.06 |
| Evo 1.5 | -0.01 | -0.05 |
| Nucleotide Transformer v3 (650M) | 0.05 | -0.03 |
| Nucleotide Transformer v3 (8M) | 0.04 | 0.03 |
| Nucleotide Transformer v3 (100M) | 0.10 | -0.01 |
| AIDO.RNA (1M) | 0.07 | -0.06 |
| Orthrus | 0.13 | 0.09 |
| Evo 1 | 0.00 | -0.02 |
| GenSLM (2.5B) | 0.10 | 0.01 |
<!-- END GENERATED TABLES -->

[Full precision CSV][0]. Small differences should be read cautiously, especially
with only two tRNA and two aptamer molecules. Coding and splicing assays are
excluded because they measure protein function or splicing efficiency.

## Scoring

Masked language models use `wt-fill`: mask each mutated position in the wild-type
sequence and sum `log p(mutant) - log p(wild type)`. This follows the [ESM][1]
and [ProteinGym][2] implementations. All four [fill strategies][3] are available
in the prediction files. RNA-ERNIE uses unmasked wild-type probability changes.
GenSLM uses mean next-codon log likelihood, excluding padding. EVmutation
scores the change in statistical energy of a pairwise model fitted to each
assay's Riboseek MSA. RNA-2Di sums, over the dinucleotides a variant changes,
the [substitution log odds][7] of Riboseek's dinucleotide matrix file relative to
the wild type's own score. Mutation count is the
negative number of substitutions.

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
[7]: https://github.com/steineggerlab/riboseek/blob/ef28e4bd0d023f683d9ba40a871a99baecb2bfcf/data/dinuc.out
[8]: leaderboard_by_mutation_count.csv
