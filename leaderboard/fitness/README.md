# Fitness leaderboard

Signed Spearman correlation with experimental fitness on 31 ncRNA assays.
Columns average assays within each category. Macro weights the three categories
equally. Higher is better.

| Rank | Model | Ribozyme (n=26) | tRNA (n=3) | Aptamer (n=2) | Macro (3 ncRNA) |
|---:|:--|--:|--:|--:|--:|
| 1 | AIDO.RNA (650M) | 0.0660 | 0.4894 | 0.0934 | 0.2163 |
| 2 | Evo 2 (20B) | 0.1105 | 0.4345 | 0.0937 | 0.2129 |
| 3 | Evo 2 (40B) | 0.1081 | 0.4310 | 0.0970 | 0.2120 |
| 4 | AIDO.RNA (1.6B) | 0.0609 | 0.4884 | 0.0718 | 0.2070 |
| 5 | Evo 2 (1B base) | 0.0087 | 0.4466 | 0.1334 | 0.1962 |
| 6 | RNA-ERNIE | 0.1343 | 0.4161 | 0.0306 | 0.1937 |
| 7 | Evo 2 (7B) | 0.0651 | 0.3867 | 0.1192 | 0.1904 |
| 8 | RNAGenesis | 0.0750 | 0.4381 | 0.0343 | 0.1825 |
| 9 | RiNALMo | -0.0243 | 0.4856 | 0.0459 | 0.1690 |
| 10 | AIDO.RNA (300M) | -0.0256 | 0.4551 | 0.0372 | 0.1556 |
| 11 | AIDO.RNA (25M) | -0.0299 | 0.4569 | 0.0348 | 0.1540 |
| 12 | Evo 1.5 | 0.0278 | 0.3850 | 0.0007 | 0.1378 |
| 13 | Nucleotide Transformer v3 (8M) | -0.0111 | 0.3053 | 0.0988 | 0.1310 |
| 14 | RNA-FM | -0.0225 | 0.4147 | -0.0043 | 0.1293 |
| 15 | Nucleotide Transformer v3 (650M) | -0.0240 | 0.3518 | 0.0436 | 0.1238 |
| 16 | Nucleotide Transformer v3 (100M) | -0.0174 | 0.2544 | 0.1001 | 0.1124 |
| 17 | AIDO.RNA (1M) | 0.0014 | 0.1777 | 0.0626 | 0.0806 |
| 18 | Orthrus | -0.0349 | 0.0922 | 0.1578 | 0.0717 |
| 19 | Evo 1 | -0.0216 | 0.0948 | 0.0058 | 0.0263 |
| 20 | GenSLM | -0.0045 | -0.0934 | -0.0036 | -0.0338 |

[Full precision CSV][0]. Small differences should be read cautiously, especially
with only three tRNA and two aptamer assays. Coding and splicing assays are
excluded because they measure protein function or splicing efficiency.

## Scoring

Masked language models use `wt-fill`: mask each mutated position in the wild-type
sequence and sum `log p(mutant) - log p(wild type)`. This follows the [ESM][1]
and [ProteinGym][2] implementations. All four [fill strategies][3] are available
in the prediction files. RNA-ERNIE uses unmasked wild-type probability changes.

Evo2 FP8 scores can vary slightly across GPUs.

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
