# RNAGym secondary structure benchmark

## Getting started

1. The datasets are stored under [`../../data/2d/`](../../data/2d/). See the
[repository-level instructions](../../README.md#resources) to download and extract
them.
2. Set machine-specific paths under `[activation.env]` in `pixi.toml`.

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

## Scoring

Chemical mapping is scored as cluster-macro Spearman between finite reactivities
(most NaNs are unmeasured construct regions such as barcodes) and predicted
residue unpaired probabilities (`1 − Σj Pij`). DMS is scored over A/C, CMCT
over G/U, and other modalities over all bases. Spearman avoids directly
comparing reactivity magnitudes with probabilities or choosing an arbitrary
classification threshold. Constant predictions receive zero.

> [!NOTE]
> For neural models, summed pair probabilities may exceed one at a residue.
> Following
> [Arnie's official RibonanzaNet
> inference](https://github.com/WaymentSteeleLab/arnie/blob/660de8139bd2198bbe115adadd5bc5f12183f9f4/src/arnie/pk_predictors.py#L111-L116)
> highly confident residues are clipped to 1 before conversion.

Discrete structures are scored as cluster-macro F1 using each model's official
decoders. PDB pairs touching unresolved residues are excluded. Unsupported
sequences receive zero.
