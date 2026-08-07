# RNAGym secondary structure benchmark

## Getting started

The datasets are stored under [`../../data/2d/`](../../data/2d/). See the
[repository-level instructions](../../README.md#resources) to download and extract
them.

## Environments

This directory has isolated [Pixi](https://pixi.sh) environments for the
secondary structure baselines in RNAGym:

| Environment | Tool |
| --- | --- |
| `(default)` | Default env for scripts and processing
| `ribonanzanet` | RibonanzaNet
| `eternafold` | EternaFold
| `contrafold` | CONTRAfold
| `vienna` | ViennaRNA
| `rnastructure` | RNAstructure
| `rna-fm` | RNA-FM
| `ufold` | UFold
| `mxfold2` | MXfold2

Install the environments with `pixi install --all`, then enter an environment
with `pixi shell -e <name>`, or run a command directly with `pixi run -e <name>
<command>`.

### Note on source-only neural models

RibonanzaNet and UFold are not packaged by their authors. Their `source` task
checks out a pinned upstream commit under `.pixi/model-sources/`; `test` runs
this task automatically. For example:

```bash
pixi run -e ufold source
pixi run -e ufold test
```

Neural-model tests fetch verified public weights automatically and fold one
short sequence on CPU. Use a GPU node for practical inference.

## Reproducing the predictions

After verifying the contents of `sh/predict.sh` for your cluster, generate
predictions with a model and save them to
`../../data/2d/predictions/<model>` with:

```bash
pixi run -e <env> predict
```

To generate the leaderboard from the saved predictions:

```bash
pixi run leaderboard
```

This writes the [2D leaderboard](../../leaderboard/2d/README.md) and its
[detailed scores](../../leaderboard/2d/leaderboard.csv).
