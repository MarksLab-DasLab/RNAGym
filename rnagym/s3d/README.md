# RNAGym 3D structure benchmark

This directory includes the code and environments used by RNAGym for 3D
structure prediction. The datasets are stored under
[`../../data/3d/`](../../data/3d/).

## Datasets

RNAGym provides quality-filtered PDB evaluation targets released after the
latest baseline training cutoff. Targets are stored in
`data/3d/rnagym_3d.parquet` with a `type` of `monomer` or `multimer`.

`data/3d/curation/annotated_chains.csv` contains all RNA chains >16nt with
useful annotations like Rfams hits, heteroatoms bound, % bound by polymer,
resolution, and much more. This can be used to easily filter the RNAs in the
PDB by different selection criteria, for example identifying self-structured
RNA monomers in the PDB that bind ligands.

### Data Sources

RNA chain candidates were collected from [RNA3DB][22] prior to
processing and filtering. Where appropriate, overlaps with the following
datasets were noted:

| Dataset                                                                                              | # of PDBs |
| ---------------------------------------------------------------------------------------------------- | --------- |
| [RNA3DB][22]                                                                                         | 23816     |
| [RNA 3D Hub][4]                                                                                      | 23027     |
| [Evaluating DCA-based method performances for RNA contact prediction by a well-curated data set][1]  | 68        |
| [RNA Puzzles][2]                                                                                     | 49        |
| [CASP 15 (RNA category)][3]                                                                          | 13        |
| [A Comparative Review of Deep Learning Methods for RNA Tertiary Structure Prediction][14]            | 164       |

## Generating the datasets

To run the RNAGym dataset curation pipeline:

1. Modify [`rnagym/config.py`](../config.py) to suit your system
2. Run `pixi install`
3. Set `RNAGYM_DATABASE_DIR` to the directory containing `Rfam-15.0/` and run
   `pixi run pipeline` to generate the processed files under `data/3d/`

```bash
RNAGYM_DATABASE_DIR=/path/to/databases pixi run pipeline
```

The default workflow is as follows:

1. `merge`: Merges the initial datasets into `curation/merged_pdb_ids.csv`, a
   list of unique PDB IDs containing RNA chains.
2. `annotate`: Individual RNA chains are annotated into
   `curation/annotated_chains.csv`.
3. `split`: Selects the monomer and multimer targets in `rnagym_3d.parquet`.
   - To determine the best split, RNAGym calculates the maximal TM score
     between each candidate chain and any chain from the baseline
     training sets. This requires about 5-60 minutes per CPU per
     candidate chain.  To speed things up, you can launch Slurm jobs for
     all these tasks using `pixi run tm-train`.  The results will
     be cached for the next `split`.

You can run any step individually with `pixi run <command>`.

### Training-structure comparisons

`pixi run usalign` launches a 64-task Slurm array for missing target-to-training
comparisons. Completed outputs are reused, and rerunning `pixi run split` adds
the resulting annotations to `rnagym_3d.parquet`.

The shared data archive described in the [root README](../../README.md) includes
the precomputed 3D US-align outputs under `data/3d/usalign/`.

## Launching RNA predictions

RNAGym currently evaluates the following baselines:

1. AlphaFold3
2. NuFold
3. RhoFold+
4. RoseTTAFold2NA
5. trRosettaRNA

To launch the predictions, you can:

1. Once per database release, run `pixi run riboseek-db` to install
   [Riboseek][21] 1.0.0 and prepare RNAcentral 27.0 and full NCBI nt under
   `RNAGYM_DATABASE_DIR`. Full nt requires several terabytes of storage.
2. Generate one A3M and aligned FASTA for each unique 3D or ncRNA fitness
   sequence with `pixi run riboseek`. RNAcentral is searched forward-only and
   nt on both strands using four GPUs. The combined hits seed a covariance
   model that realigns them before MSA construction. Alignments are stored
   under each benchmark's directory in `data/`.
3. Launch your predictions using `pixi run predict <baseline>`.
    - Valid baselines are currently `af3`, `nufold`, `rhofold`, `rf2na`,
      and `trRNA`.
    - Before running, configure `./scripts/<baseline>.sh` for your Slurm
      cluster and point it to a valid installation of the respective
      baseline.

## Analysis

Generate the [3D leaderboard](../../leaderboard/3d/) and detailed scores with:

```bash
pixi run leaderboard
```

Full structural analysis can be conducted using `pixi run evcouplings`
followed by `pixi run analyze`.

<!--Hyperlinks-->
[1]: https://pubmed.ncbi.nlm.nih.gov/32276988/
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7145511/
[3]: https://predictioncenter.org/casp15/results.cgi?tr_type=rna
[4]: http://www.springerlink.com/content/u54511012r0344h3/
[5]: https://rna.bgsu.edu/rna3dhub/nrlist/
[6]: https://www.rcsb.org
[7]: https://www.bgsu.edu/research/rna/software/fr3d.html
[8]: https://www.bgsu.edu/research/rna/web-applications/jar3d.html
[9]: https://nakb.org/ndbmodule/bp-catalog/
[10]: https://github.com/murfalo/evcouplings
[11]: https://github.com/BGSU-RNA/fr3d-python
[12]: https://docs.google.com/spreadsheets/d/1AORpL9zm9m-Tvdw5xvg7cbyo4C-blKD5
[14]: https://www.biorxiv.org/content/10.1101/2024.11.27.625779v1
[15]: https://github.com/marcellszi/rna3db
[16]: https://docs.google.com/spreadsheets/d/1AORpL9zm9m-Tvdw5xvg7cbyo4C-blKD5/edit?gid=1076929196#gid=1076929196&fvid=329186721
[17]: https://docs.google.com/spreadsheets/d/1AORpL9zm9m-Tvdw5xvg7cbyo4C-blKD5/edit?gid=1076929196#gid=1076929196&fvid=1703988487
[18]: https://docs.google.com/spreadsheets/d/1AORpL9zm9m-Tvdw5xvg7cbyo4C-blKD5/edit?gid=1555832897#gid=1555832897
[19]: https://github.com/marcellszi/rna3db/tree/main
[20]: https://github.com/marcellszi/rna3db/releases/tag/2024-12-04-full-release
[21]: https://github.com/steineggerlab/riboseek
[22]: https://github.com/marcellszi/rna3db
[23]: https://github.com/RNA-Puzzles/RNA_assessment
