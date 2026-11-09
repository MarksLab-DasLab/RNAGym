# Clinical test fixture

Real variants and published scores for the workflow test in
`tests/test_clinical.py`, small enough to evaluate in seconds.

| Path | Contents |
| --- | --- |
| `staged/<file>.parquet` | The staged columns the evaluation reads, for a sample of rows (no sequences) |
| `scores/<file>.parquet` | `row_id` and one score column per model, the published scores of those rows |
| `expected/<file>.groups.csv`, `expected/<file>.headline.csv` | The published evaluation code's tables for this fixture, 20 bootstrap draws, seed 20261008 |
| `expected/leaderboard.csv`, `units.csv`, `classes.csv`, `mito_strata.csv` | The leaderboard tables of that evaluation, computed by a separate script with the published figures' formulas |
| `leaderboard.md` | The README the leaderboard writes for this fixture, checked cell by cell against the expected tables when it was made |

Rows per nuclear file: the three consequence classes with the most of their
scarcer label give 12 pathogenic and 12 benign variants each, the next class 4
and 4 (below the floor of 10), up to 4 `unknown` variants, up to 6 variants a
model could not score, and for pre-mRNA SNVs 10 pathogenic and 10 benign missense
variants of one gene (a gene macro). The two mitochondrial files are whole, so
their AUROCs equal the published ones.
