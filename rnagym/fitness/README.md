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

GenSLM reports mean next-codon log likelihood, excluding padding. Masked models
compute all four [fill strategies][4], with `wt-fill` used for the leaderboard.

## Quality checks

```bash
pixi run test
pixi run lint
pixi run --locked -e default check-published
```

The fast tests use real assay fixtures and need no checkpoints. The full check
rebuilds both tables and scores 128 variants from each of the 31 ncRNA assays
with all 20 checkpoints. Score ranks must have Spearman correlation of at least
0.95 with the stored predictions. Install environments and weights first.
The full run has a one-hour limit and needs two 80 GB GPUs for Evo2 40B.

Use `pixi run -e <env> check-published` to check one family, or add
`--leaderboard-only` to the default command to check aggregates.

Run `pixi run coverage` before merging. It runs the tests and full reproduction
and requires 80% coverage.

[0]: ../../leaderboard/fitness/
[1]: ../../README.md#getting-started
[2]: pixi.toml
[3]: https://g-e71d1.fd635.8443.data.globus.org/models/2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt
[4]: baselines/masked_lm/README.md
