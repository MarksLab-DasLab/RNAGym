# RNAGym secondary structure benchmark

## Data setup

From this directory, download and extract the data archive:

```bash
wget https://marks.hms.harvard.edu/rnagym/structure_prediction/rnagym_2d.tar.xz
tar -xJf rnagym_2d.tar.xz
```
## Environments

This directory has isolated [Pixi](https://pixi.sh) environments for the
secondary structure baselines in RNAGym:

| Environment | Tool |
| --- | --- |
| `(default)` | Default env for scripts and processing
| `ribonanzanet` | RibonanzaNet
| `arnie` | EternaFold, CONTRAfold, ViennaRNA, RNAstructure
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

To generate predictions with a model and save them to
`data/chemical_mapping/predictions/`:

<!-- TODO(MCA): Implement -->
```bash
pixi run -e <env> predict
```

<!-- TODO(MCA): Implement -->
To generate the leaderboard from the saved predictions:

```bash
pixi run leaderboard
```

### How models are scored

For each test profile, pair probabilities are converted to unpaired probabilities
(`1 − ΣPij`), and only positions with finite reactivities are scored (the vast
majority of NaNs are fixed, unmeasured construct regions like barcodes).
Per-profile Spearman, AUC, and F1 are computed and macro-averaged across profiles
and reagents.

<-- TODO(MCA): Add PseudoBase scoring description -->
