# Fitness benchmark

RNA fitness predictions evaluated against 70 experimental assays. The
[leaderboard][0] uses signed Spearman correlation on 31 ncRNA assays, averaged
within ribozyme, tRNA and aptamer categories, then equally across categories.

## Setup

From the repository root, download the [assays and predictions][1]:

```bash
cd data/fitness
wget https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_processed_assays.zip
wget https://marks.hms.harvard.edu/rnagym/fitness_prediction/model_predictions.zip
unzip fitness_processed_assays.zip -d assays
unzip model_predictions.zip
cd ../../rnagym/fitness
pixi install
```

Data defaults to `data/fitness/`. Set `RNAGYM_DATA_DIR` to another parent data
directory if needed. Checkpoints are shared under `/n/lw_groups/marks/ckpt/<model>`.
Set `RNAGYM_CHECKPOINT_DIR` to use another location.

GenSLM 2.5B, Evo 1 and Evo 1.5 were regenerated on September 6, 2026.
The public prediction archive still contains their earlier scores.

## Run

From `rnagym/fitness/`:

```bash
pixi run merge
pixi run leaderboard
SLURM_ARRAY_TASK_ID=12 pixi run -e ntv3 predict
```

`merge` and `leaderboard` default to ncRNA. Use `--models` to select models.
`--type all` includes coding and splicing assays and requires predictions for them.
Results go to `data/fitness/reports/`.
Repeated experimental measurements are retained. Missing or conflicting
predictions fail the merge.

`predict` scores one reference-sheet row, defaulting to row 0. Choose a model
[environment][2] with `-e`. Sources and public weights are fetched automatically.
Gated checkpoints require access. Download the [GenSLM 2.5B checkpoint][3] to
`genslm/2.5B/` under the checkpoint root. EVmutation uses Riboseek MSAs under
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
rebuilds the leaderboard, checks native GenSLM and EVmutation scoring, then
reruns all 20 checkpoints on 128 variants from each of the 31 ncRNA assays.
It requires matching rows and missing values, finite scores, and old/new score
Spearman of at least 0.95. Score-magnitude and fitness-correlation differences
are reported separately. Install environments and weights first. The full run
has a shared one-hour deadline and needs two 80 GB GPUs for Evo2 40B.

Use `pixi run -e ntv3 check-published` to check one family, or add
`--leaderboard-only` to the default command to check aggregates. Reports,
including failures, go to the temporary directory. Use `--report PATH` to choose
the report location. Acceptance bounds are fixed in the [checker][5].

`pixi run coverage` runs the fast tests and full reproduction, requiring 80%
coverage. Full reproduction remains a required check before merging.

[0]: ../../leaderboard/fitness/
[1]: https://marks.hms.harvard.edu/rnagym/fitness_prediction
[2]: pixi.toml
[3]: https://g-e71d1.fd635.8443.data.globus.org/models/2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt
[4]: baselines/masked_lm/README.md
[5]: tasks/check_published.py
