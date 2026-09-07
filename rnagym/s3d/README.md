# Tertiary structure benchmark

RNA 3D structure prediction on quality-filtered targets released after
2023-01-12 in the [RNA3DB][0] 2026-01-05 release. Repeated sequences with distinct
structures are retained. Predictions and MSAs are shared by sequence.
See the [leaderboard][1] for results.

## Prepare data

Configure the data and database paths in [Config3D][2], then run here:

```bash
pixi run pipeline
pixi run usalign
pixi run split
```

`pipeline` downloads and filters RNA3DB. It requires Rfam 15.0 with a pressed
`Rfam.cm` and `family.txt.gz`. `usalign` compares targets against each model's
training structures. `split` adds those comparisons to `data/3d/rnagym_3d.parquet`.

## Predict and evaluate

```bash
pixi run riboseek-db
pixi run riboseek
pixi run -e af3 predict
pixi run score
pixi run leaderboard
```

Choose a model environment from [pixi.toml][3]. Sources and public checkpoints
are fetched automatically. AlphaFold 3 requires licensed parameters and its
databases. RF2NA requires PDB100 and must run after AF3 to reuse partner-chain
MSAs. [Riboseek][4] databases live under `RNAGYM_DATABASE_DIR`.

Checkpoints are shared under `/n/lw_groups/marks/ckpt/<model>`.
Set `RNAGYM_CHECKPOINT_DIR` to use another root. Predictions and scores go to
`data/3d/`, and `leaderboard` updates the leaderboard CSV and README.

[0]: https://github.com/marcellszi/rna3db/releases/tag/2026-01-05-full-release
[1]: ../../leaderboard/3d/
[2]: ../config.py
[3]: pixi.toml
[4]: https://github.com/steineggerlab/riboseek
