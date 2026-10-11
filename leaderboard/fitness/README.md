# Fitness leaderboard

Signed Spearman correlation with experimental fitness on 29 ncRNA assays of 27
RNA molecules. Assays of the same molecule are averaged first, as ProteinGym
averages the assays of one protein. Columns then average molecules within each
category, and Macro weights the three categories equally. Depth macro instead
computes Spearman separately at each mutation count, then averages. This
measures how well models rank variants when the naive prior that more mutations
are worse (see Mutation count) no longer helps. Italic rows are simple baselines
without pretraining. Higher is better.

<!-- BEGIN GENERATED TABLES -->
| Rank | Model | Ribozyme (n=24) | tRNA (n=3) | Aptamer (n=2) | Macro (3 ncRNA) | Depth macro |
|---:|:--|--:|--:|--:|--:|--:|
| 1 | _EVmutation_\* | 0.25 | 0.44 | 0.14 | 0.27 | 0.14 |
| 2 | _RNA-2Di_ | 0.25 | 0.47 | -0.07 | 0.22 | 0.12 |
| 3 | AIDO.RNA (650M) | 0.06 | 0.47 | 0.09 | 0.21 | 0.07 |
| 4 | Evo 2 (20B) | 0.10 | 0.42 | 0.09 | 0.21 | 0.09 |
| 5 | Evo 2 (40B) | 0.10 | 0.42 | 0.10 | 0.21 | 0.09 |
| 6 | AIDO.RNA (1.6B) | 0.05 | 0.47 | 0.07 | 0.19 | 0.06 |
| 7 | Evo 2 (7B) | 0.06 | 0.38 | 0.12 | 0.19 | 0.07 |
| 8 | Evo 2 (1B base) | 0.00 | 0.42 | 0.13 | 0.18 | 0.07 |
| 9 | _Mutation count_ | 0.20 | 0.46 | -0.12 | 0.18 | 0.00 |
| 10 | RNA-ERNIE | 0.14 | 0.38 | 0.03 | 0.18 | 0.10 |
| 11 | RNAGenesis | 0.07 | 0.43 | 0.03 | 0.18 | 0.03 |
| 12 | RiNALMo | -0.02 | 0.47 | 0.05 | 0.17 | 0.05 |
| 13 | AIDO.RNA (300M) | -0.02 | 0.43 | 0.04 | 0.15 | 0.03 |
| 14 | AIDO.RNA (25M) | -0.02 | 0.43 | 0.03 | 0.15 | 0.04 |
| 15 | Evo 1.5 | 0.06 | 0.35 | 0.00 | 0.14 | 0.05 |
| 16 | Nucleotide Transformer v3 (8M) | -0.01 | 0.27 | 0.10 | 0.12 | 0.05 |
| 17 | RNA-FM | -0.01 | 0.37 | 0.00 | 0.12 | -0.01 |
| 18 | Nucleotide Transformer v3 (650M) | -0.02 | 0.33 | 0.04 | 0.12 | 0.03 |
| 19 | Nucleotide Transformer v3 (100M) | -0.01 | 0.24 | 0.10 | 0.11 | 0.05 |
| 20 | AIDO.RNA (1M) | 0.01 | 0.14 | 0.06 | 0.07 | 0.04 |
| 21 | Orthrus | -0.04 | 0.08 | 0.16 | 0.06 | 0.05 |
| 22 | Evo 1 | -0.01 | 0.09 | 0.03 | 0.04 | -0.01 |
| 23 | GenSLM (2.5B) | -0.02 | 0.05 | 0.06 | 0.03 | 0.03 |

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
GenSLM uses mean next-codon log likelihood, excluding padding. EVmutation
scores the change in statistical energy of a pairwise model fitted to each
assay's Riboseek MSA. RNA-2Di sums Riboseek's dinucleotide [substitution log
odds][7] over the dinucleotides a variant changes. Mutation count is the
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
