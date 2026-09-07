# Fitness benchmark

RNA fitness predictions evaluated against 70 experimental assays. The
[leaderboard][0] uses signed Spearman correlation on 31 ncRNA assays, averaged
within ribozyme, tRNA and aptamer categories, then equally across categories.

## Setup

Complete the [shared data setup][1], then run from the repository root:

```bash
cd rnagym/fitness
pixi install
```

Fitness data lives under `data/fitness/`.

## Run

From `rnagym/fitness/`:

```bash
pixi run -e <env> predict
```

Replace `<env>` with a model environment from [pixi.toml][2]:

| Environment | Model |
| --- | --- |
| `aido-rna` | AIDO.RNA |
| `evmutation` | EVmutation |
| `evo` | Evo 1 / 1.5 |
| `evo2` | Evo 2 |
| `genslm` | GenSLM |
| `ntv3` | Nucleotide Transformer v3 |
| `orthrus` | Orthrus |
| `rna-ernie` | RNA-ERNIE |
| `rna-fm` | RNA-FM |
| `rnagenesis` | RNAGenesis |
| `rinalmo` | RiNALMo |

`predict` defaults to all assays. To select reference-sheet rows, append an
index or range, for example `predict 12` or `predict 0-8,12`. Slurm array jobs
use `SLURM_ARRAY_TASK_ID` unless a selection is passed explicitly.

To evaluate saved predictions:

```bash
pixi run merge
pixi run leaderboard
```

`leaderboard` writes both ncRNA tables under `leaderboard/fitness/`. The
supplemental comparison uses the variants scored by EVmutation. Repeated
experimental measurements are retained. Incomplete or conflicting predictions
fail the merge.

Sources and weights are fetched automatically, except for [GenSLM 2.5B][3].
Download its checkpoint through authenticated Globus access to
`$RNAGYM_CHECKPOINT_DIR/genslm/2.5B/`. EVmutation uses Riboseek MSAs under
`data/fitness/msa/by_assay/` and leaves uncovered variants unscored.

Evo2 FP8 scores can vary slightly across GPUs. GenSLM, Evo1 and Evo1.5 use
float32 for reproducibility. GenSLM reports mean next-codon log likelihood,
excluding padding. Masked models compute all four [fill strategies][4], with
`wt-fill` used for the leaderboard.

## Quality checks

```bash
pixi run test
pixi run lint
pixi run --locked -e default check-published
```

The fast tests use real assay fixtures and need no checkpoints. The full check
rebuilds the leaderboard and reruns all 20 published checkpoints on 128 variants
from each of the 31 ncRNA assays.
It requires matching rows and missing values, finite scores, and old/new score
Spearman of at least 0.95. Score-magnitude and fitness-correlation differences
are reported separately. Install environments and weights first. The full run
has a shared one-hour deadline and needs two 80 GB GPUs for Evo2 40B.

Use `pixi run -e <env> check-published` to check one family, or add
`--leaderboard-only` to the default command to check aggregates. Acceptance
bounds are fixed in the [checker][5].

`pixi run coverage` runs the fast tests and full reproduction, requiring 80%
coverage. Full reproduction remains a required check before merging.

[0]: ../../leaderboard/fitness/
[1]: ../../README.md#getting-started
[2]: pixi.toml
[3]: https://g-e71d1.fd635.8443.data.globus.org/models/2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt
[4]: baselines/masked_lm/README.md
[5]: tasks/check_published.py
