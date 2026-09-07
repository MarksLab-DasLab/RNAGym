# Secondary structure benchmark

RNA secondary structure prediction on the RNAGym datasets in `data/2d/`.
See the [leaderboard][0] for results.

## Run

Complete the [shared data setup][3], then run from `rnagym/s2d/`:

```bash
pixi install
pixi run -e <env> test
pixi run -e <env> predict
```

Replace `<env>` with a model environment from [pixi.toml][1]:

| Environment | Model |
| --- | --- |
| `contrafold` | CONTRAfold |
| `eternafold` | EternaFold |
| `mxfold2` | MXfold2 |
| `ribonanzanet` | RibonanzaNet |
| `rinalmo` | RiNALMo |
| `rna-fm` | RNA-FM |
| `rnastructure` | RNAstructure |
| `ufold` | UFold |
| `vienna` | ViennaRNA |

Model sources and public weights are fetched automatically. Checkpoints use
`RNAGYM_CHECKPOINT_DIR`. RiNALMo tests require a GPU.

Prediction jobs use the Slurm settings in [sh/predict.sh][2]. Outputs go to
`data/2d/predictions/<model>/`. Once the jobs finish, run `pixi run leaderboard`
to score the predictions and update the leaderboard CSV and README.

[0]: ../../leaderboard/2d/
[1]: pixi.toml
[2]: sh/predict.sh
[3]: ../../README.md#getting-started
