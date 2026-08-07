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

This writes `../../leaderboard/2d/leaderboard.csv`. EternaFold and RibonanzaNet
are marked with `*` for chemical mapping because their training data overlap
the benchmark.

### How models are scored

Scores use all five folds.

For chemical mapping, residue unpaired probabilities (`1 − Σj Pij`) are
predicted and compared with experimental reactivities using Spearman. DMS is
scored over A/C, CMCT over G/U, and other modalities over all bases. Scores are
calculated over finite reactivities (the vast majority of NaNs are fixed,
unmeasured construct regions like barcodes) for each profile, then averaged
within each sequence cluster and across clusters. Each modifier, including each
degradation condition, is reported separately. Spearman avoids directly
comparing reactivity magnitudes with probabilities or imposing an arbitrary
reactivity threshold for classification. Constant predictions receive a score
of zero.

> [!NOTE]
> Neural models generally treat each i,j pair as an independent prediction,
> with no/limited constraint on column probabilities. When neural models assign
> residue pair probabilities above one, we follow [Arnie's official
> RibonanzaNet
> inference](https://github.com/WaymentSteeleLab/arnie/blob/660de8139bd2198bbe115adadd5bc5f12183f9f4/src/arnie/pk_predictors.py#L111-L116)
> and clip the sum to `[0, 1]` before conversion.  This leaves most residues
> intact, but clips highly confident residues to 1.

For discrete structures from PseudoBase, the PDB, or bpRNA-1m, each model's
official decoder is used (Hungarian, MFE, MEA, Viterbi, etc. as applicable). F1
is computed between each reference structure and the model-decoded structure,
then averaged within each sequence cluster and across clusters. PDB pairs
touching unresolved residues are excluded. Unsupported sequences receive F1 0.
