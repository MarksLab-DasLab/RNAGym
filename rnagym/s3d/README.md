# RNAGym 3D structure benchmark

This directory includes the code and environments used by RNAGym for 3D
structure prediction. The datasets are stored under
[`../../data/3d/`](../../data/3d/).

## Datasets

RNAGym draws candidate chains from the [RNA3DB][22] 2025-10-01 release. The
benchmark retains quality-filtered PDB targets released after 2023-01-12, the
PDB snapshot date used by the original RNAGym split. Model-specific training
homology uses each model's structural training cutoff, including 2021-09-30 for
AF3. Targets are stored in
`data/3d/rnagym_3d.parquet` with a `type` of `monomer` or `multimer`. Repeated
sequences with distinct experimental structures are retained. `sequence_id`
identifies exact sequences, and `cluster_rep` identifies the shared 40%
sequence identity clusters. Monomer predictions and MSAs are generated once
per `sequence_id`, then scored against every corresponding structure.

`data/3d/curation/annotated_chains.csv` contains RNA3DB chains of at least 8 nt
with Rfam hits, bound heteroatoms, polymer coverage, resolution, and other
filtering metadata. For example, it can identify self-structured RNA monomers
that bind ligands.

## Generating the datasets

Configure `Config3D` in [`rnagym/config.py`](../config.py), then reconstruct the
dataset from this directory with:

```bash
pixi run pipeline
```

Rfam 15.0 must contain a pressed `Rfam.cm` and `family.txt.gz` at the paths
defined by `Config3D`.

The default workflow is as follows:

1. `annotate`: Individual RNA3DB chains are annotated into
   `curation/annotated_chains.csv`.
2. `split`: Selects every quality-filtered post-cutoff monomer and multimer,
   then updates the shared sequence registry and 40% identity clusters.

You can run any step individually with `pixi run <command>`.

### Training-structure comparisons

`pixi run usalign` launches a 64-task Slurm array for missing target-to-training
comparisons. Completed outputs are reused, and rerunning `pixi run split` adds
the resulting annotations to `rnagym_3d.parquet`.

The shared data archive described in the [root README](../../README.md) includes
the precomputed 3D US-align outputs under `data/3d/usalign/`.

## Launching RNA predictions

RNAGym currently evaluates the following baselines:

1. AlphaFold 3
2. NuFold
3. RhoFold+
4. RoseTTAFold2NA
5. trRosettaRNA

To launch the predictions, you can:

1. Once per database release, run `pixi run riboseek-db` to install
   [Riboseek][21] 1.0.1 and prepare RNAcentral 27.0 and full NCBI nt under
   `RNAGYM_DATABASE_DIR`. Full nt requires several terabytes of storage.
2. Generate one A3M and aligned FASTA for each unique 3D or ncRNA fitness
   sequence with `pixi run riboseek`. RNAcentral is searched forward-only and
   nt on both strands using four GPUs. The combined hits seed a covariance
   model that realigns them before MSA construction. Alignments are stored
   under each benchmark's directory in `data/`.
3. Launch your predictions using `pixi run -e <env> predict`. Run RF2NA after
   AF3 because it reuses the partner-chain MSAs prepared by AF3.

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
[22]: https://github.com/marcellszi/rna3db
