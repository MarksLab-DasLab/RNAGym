# RNAGym 3D structure benchmark

This directory includes the code and environments used by RNAGym for 3D
structure prediction. The datasets are stored under
[`../../data/3d/`](../../data/3d/).

## Datasets

RNAGym retains every quality-filtered target released after 2023-01-12 from the
[RNA3DB][22] 2026-01-05 full release. Repeated sequences with distinct
structures are retained, while predictions and MSAs are generated once per
unique sequence. Model-specific training homology uses each model's structural
training cutoff.

`data/3d/curation/annotated_chains.parquet` contains RNA3DB chains of at least 8
nt and the Rfam, polymer coverage, resolution, and other metadata used to filter
them.

## Generating the datasets

Configure `Config3D` in [`rnagym/config.py`](../config.py), then reconstruct the
dataset from this directory with:

```bash
pixi run pipeline
```

This downloads the pinned RNA3DB release, annotates its chains, selects the
targets, and updates the shared sequence registry. Rfam 15.0 must contain a
pressed `Rfam.cm` and `family.txt.gz` at the paths defined by `Config3D`. Each
step can also be run individually with `pixi run <command>`.

### Training-structure comparisons

`pixi run usalign` launches the missing target-to-training comparisons.
Completed outputs are reused. Run `pixi run split` afterward to add the results
to `rnagym_3d.parquet`.

The shared data archive described in the [root README](../../README.md) includes
the precomputed 3D US-align outputs under `data/3d/usalign/`.

## Launching RNA predictions

RNAGym currently evaluates the following baselines:

1. AlphaFold 3
2. NuFold
3. RhoFold+
4. RoseTTAFold2NA
5. trRosettaRNA

To launch predictions:

1. Run `pixi run riboseek-db` once to prepare [Riboseek][21], RNAcentral, and
   NCBI nt under `RNAGYM_DATABASE_DIR`.
2. Run `pixi run riboseek` to generate the MSAs.
3. Run `pixi run -e <env> predict` for each model. Run RF2NA after AF3 because
   it reuses AF3's partner-chain MSAs.

Public model sources and checkpoints are installed automatically. AlphaFold 3
requires its licensed parameters and databases, while RF2NA requires its
official PDB100 database. Their expected locations are defined by `Config3D`.

## Analysis

Score every available model and generate the
[3D leaderboard](../../leaderboard/3d/) with:

```bash
pixi run score
pixi run leaderboard
```

<!--Hyperlinks-->
[21]: https://github.com/steineggerlab/riboseek
[22]: https://github.com/marcellszi/rna3db/releases/tag/2026-01-05-full-release
