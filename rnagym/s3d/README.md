# Tertiary structure benchmark

RNA 3D structure prediction on quality-filtered targets released after
2023-01-12 in the [RNA3DB][0] 2026-01-05 release. Repeated sequences with distinct
structures are retained. Predictions and MSAs are shared by sequence.
See the [leaderboard][1] for results.

## Setup

Complete the [shared data setup][5], then run from `rnagym/s3d/`:

```bash
pixi install
```

The shared archive includes the datasets, MSAs and precomputed US-align outputs
under `data/3d/`.

## Predict and evaluate

```bash
pixi run -e <env> predict
```

Replace `<env>` with a model environment from [pixi.toml][3]:

| Environment | Model |
| --- | --- |
| `af3` | AlphaFold 3 |
| `boltz` | Boltz-2 |
| `nufold` | NuFold |
| `of3` | OpenFold3 |
| `protenix` | Protenix-v1 |
| `rf2na` | RoseTTAFold2NA |
| `rf3` | RoseTTAFold3 |
| `rhofold` | RhoFold+ |
| `trrna` | trRosettaRNA |

Sources and public checkpoints are fetched automatically. Checkpoints use
`RNAGYM_CHECKPOINT_DIR`. AlphaFold 3 requires licensed parameters and its
databases. RF2NA requires PDB100 and must run after AF3 to reuse partner-chain
MSAs. Reference databases use `RNAGYM_DATABASE_DIR`.

`boltz`, `of3`, `protenix` and `rf3` read the target definitions and alignments
AF3 prepares, so they must run after `af3`. Each is pinned to the checkpoint
whose training cutoff predates [`TARGET_CUTOFF`][7]: `of3-p2-155k` for
OpenFold3, `protenix_base_default_v1.0.0` for Protenix and the 09/21 benchmark
checkpoint for RoseTTAFold3. Boltz-2 has no such checkpoint — its training set
runs to 2023-06-01 and so covers 128 of the 1,422 targets — and it applies
alignments to protein chains only, so its RNA chains fold from sequence alone.

Prediction jobs use the Slurm settings in [sh/predict.sh][6]. Once they finish:

```bash
pixi run score
pixi run leaderboard
```

Predictions and scores go to `data/3d/`. `leaderboard` updates the leaderboard
CSV and README.

## Rebuild data

To regenerate the shared datasets, check the paths in [Config3D][2], then run
`pixi run pipeline`. It downloads and filters RNA3DB and requires Rfam 15.0
with a pressed `Rfam.cm` and `family.txt.gz`.

`pixi run usalign` submits comparisons against each model's training structures.
Once those jobs finish, run `pixi run split` to add the results to
`data/3d/rnagym_3d.parquet`.

To regenerate MSAs, run `pixi run riboseek-db` to prepare [Riboseek][4] databases
under `RNAGYM_DATABASE_DIR`. Once those jobs finish, run `pixi run riboseek` and
wait for the alignments before predicting structures.

[0]: https://github.com/marcellszi/rna3db/releases/tag/2026-01-05-full-release
[1]: ../../leaderboard/3d/
[2]: ../config.py
[3]: pixi.toml
[4]: https://github.com/steineggerlab/riboseek
[5]: ../../README.md#getting-started
[6]: sh/predict.sh
[7]: ../config.py
