# RNAGym 3D Structure Prediction Datasets

This directory includes datasets, models, and benchmarks used by RNAGym
for 3D structure prediction.  To get started, try `./gym.py`.

## Datasets

The two datasets published with RNAGym are the eponymous `monomers.csv`
and `multimers.csv`.

### Data Sources

RNA chain candidates were collected from [RNA3DB][22]. Where
appropriate, overlaps with the following datasets were noted:

| Dataset                                                                                              | # of PDBs |
| ---------------------------------------------------------------------------------------------------- | --------- |
| [RNA3DB][22]                                                                                         | 23816     |
| [RNA 3D Hub][4]                                                                                      | 23027     |
| [Evaluating DCA-based method performances for RNA contact prediction by a well-curated data set][1]  | 68        |
| [RNA Puzzles][2]                                                                                     | 49        |
| [CASP 15 (RNA category)][3]                                                                          | 13        |
| [A Comparative Review of Deep Learning Methods for RNA Tertiary Structure Prediction][14]            | 164       |

## Generating the datasets

To generate your own editions of the RNAGym datasets:

1. Modify `util/config.py` to suit your needs
2. Install the required dependencies:
    <!--TODO(MCA): Add these to the parent environment.yml-->
    - `bioconda::infernal` (from `conda`)
    - `gemmi` (from `conda`)
    - `evcouplings` (from [Murphy's fork (private; request access)][10])
    - `hhsuite` from (`bioconda`)
    - `mmseqs=15.6` (from `conda`)
    - `rmsa` (from [source][21])
    - `rna3db` (from [source][19])
    - RNA Puzzles assessment toolkit(from [source][23])
3. Run `./gym.py` to generate `monomer.csv` and `multimer.csv`

The default workflow of `./gym.py` is as follows:

1. `merge`: Merges the initial datasets into `merged_pdb_ids.csv`, a
   list of unique PDB IDs containing RNA chains.
2. `annotate`: Individual RNA chains are annotated into `annotated_pdb_ids.csv`.
3. `split`: Splits the annotated chains into the `monomers.csv` and
   `multimers.csv` datasets published by RNAGym.
   - To determine the best split, RNAGym calculates the maximal TM score
     between each candidate chain and any chain from the baseline
     training sets. This requires about 5-60 minutes per CPU per
     candidate chain.  To speed things up, you can launch Slurm jobs for
     all these tasks using `python -m jobs.tm_train`.  The results will
     be cached for the next `split`.

You can run any of these steps individually with `./gym.py <command>`.

## Launching RNA predictions

RNAGym currently evaluates the following baselines:

1. AlphaFold3
2. NuFold
3. RhoFold+
4. RoseTTAFold2NA
5. trRosettaRNA

To launch the predictions, you can:

1. Generate MSAs with `python -m jobs.rmsa`
    - For some baselines, `.a3m` formatted MSAs are required.  You can
      use `./scripts/afa_to_a3m.sh` to generate these automatically.
2. Launch your predictions using `python -m jobs.predict <baseline>`.
    - Valid baselines are currently `af3`, `nufold`, `rhofold`, `rf2na`,
      and `trRNA`.
    - Before running, configure `./scripts/<baseline>.sh` for your Slurm
      cluster and point it to a valid installation of the respective
      baseline.

## Analysis

Analysis can be conducted using `./gym.py evcouplings` followed by
`./gym.py analyze`.  See `gym.py`, `cmd/analyze.py`, and
`cmd/evcouplings.py` for more details.

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
[13]: TBD
[14]: https://www.biorxiv.org/content/10.1101/2024.11.27.625779v1
[15]: https://github.com/marcellszi/rna3db
[16]: https://docs.google.com/spreadsheets/d/1AORpL9zm9m-Tvdw5xvg7cbyo4C-blKD5/edit?gid=1076929196#gid=1076929196&fvid=329186721
[17]: https://docs.google.com/spreadsheets/d/1AORpL9zm9m-Tvdw5xvg7cbyo4C-blKD5/edit?gid=1076929196#gid=1076929196&fvid=1703988487
[18]: https://docs.google.com/spreadsheets/d/1AORpL9zm9m-Tvdw5xvg7cbyo4C-blKD5/edit?gid=1555832897#gid=1555832897
[19]: https://github.com/marcellszi/rna3db/tree/main
[20]: https://github.com/marcellszi/rna3db/releases/tag/2024-12-04-full-release
[21]: https://github.com/pylelab/rMSA
[22]: https://github.com/marcellszi/rna3db
[23]: https://github.com/RNA-Puzzles/RNA_assessment

