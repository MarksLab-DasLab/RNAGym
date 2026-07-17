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
| `ribonanzanet` | RibonanzaNet
| `arnie` | EternaFold, CONTRAfold, ViennaRNA, RNAstructure
| `rna-fm` | RNA-FM
| `ufold` | UFold
| `mxfold2` | MXfold2

Install and test one tool at a time:

```bash
pixi install -e arnie
pixi run -e arnie test
```

Enter an environment with `pixi shell -e <name>`, or run a command directly
with `pixi run -e <name> <command>`.

### Source-only neural models

RibonanzaNet and UFold are not packaged by their authors. Their `source` task
checks out a pinned upstream commit under `.pixi/model-sources/`; `test` runs
this task automatically. For example:

```bash
pixi run -e ufold source
pixi run -e ufold test
```

Neural-model tests fetch verified public weights automatically and fold one
short sequence on CPU. Use a GPU node for practical inference.
