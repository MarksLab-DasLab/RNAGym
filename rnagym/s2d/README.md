# Secondary structure benchmark

RNA secondary structure prediction on the RNAGym datasets in `data/2d/`.
See the [leaderboard][0] for results.

## Run

From this directory, with the datasets installed:

```bash
pixi install
pixi run -e ufold test
pixi run -e ufold predict
pixi run leaderboard
```

Choose a model environment from [pixi.toml][1]. Model sources and public weights
are fetched automatically. Checkpoints are shared across benchmarks under
`/n/lw_groups/marks/ckpt/<model>`. Set `RNAGYM_CHECKPOINT_DIR` to use another root.
RiNALMo requires a GPU.

Prediction jobs use the Slurm settings in [sh/predict.sh][2]. Outputs go to
`data/2d/predictions/<model>/`. `leaderboard` scores the saved predictions and
writes the leaderboard CSV and README.

[0]: ../../leaderboard/2d/
[1]: pixi.toml
[2]: sh/predict.sh
