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

Evo2 FP8 scores can vary slightly across GPUs.
Masked models compute all four [fill strategies][4], with
`wt-fill` used for the leaderboard.

## Quality checks

```bash
pixi run test
pixi run lint
pixi run --locked -e default check-published
```

The fast tests use real assay fixtures and need no checkpoints.
`check-published` reruns the published checkpoints on one complete assay.

[0]: ../../leaderboard/fitness/
[1]: https://marks.hms.harvard.edu/rnagym/fitness_prediction
[2]: pixi.toml
[3]: https://g-e71d1.fd635.8443.data.globus.org/models/2.5B/patric_2.5b_epoch00_val_los_0.29_bias_removed.pt
[4]: baselines/masked_lm/README.md
[5]: tasks/check_published.py
