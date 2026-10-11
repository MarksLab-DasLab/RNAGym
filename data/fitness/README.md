# RNAGym fitness data

| File or directory | Contents |
| --- | --- |
| `assays/` | Processed DMS assay tables |
| `merged/` | Assays joined to model predictions |
| `model_predictions/` | One prediction directory per model |
| `msa/riboseek/` | Riboseek alignments for unique ncRNA fitness sequences |
| `msa/mmseqs/` | Conventional nucleotide MMseqs2 comparison alignments |
| `msa/riboseek/by_assay/`, `msa/mmseqs/by_assay/` | Alignment symlinks named by the paper's `DMS_ID` assay key |
| `reference_sheet_final.csv` | Assay metadata, wild-type constructs, alignment queries and the molecule each assay measures |
| `reports/` | Supplemental fitness analysis tables |

## Keeping the directory in step with the reference sheet

`assays/` holds one CSV per row of `reference_sheet_final.csv`, and
`merge_scoring_files` rejects any processed assay the sheet does not list. When
an assay is retired from the sheet, delete its `assays/` table and its
`model_predictions/*/` files before the next merge. Two assays were retired
after v0.2, `Beck_2022_ribozyme` and `Kobori_2016_osa_ribozyme`, both duplicates
of assays that remain.

## Molecules

`MOLECULE_ID` names the RNA molecule an assay measures. Assays of one molecule
are averaged before any category mean, as ProteinGym averages the assays of one
UniProt ID. `Li_2016_tRNA` and `Domingo_2018_tRNA` assay the same yeast
tRNA-Arg(CCU), and `Zhang_2024_line1_mini_ribozyme` is a 46 nt segment of
`Zhang_2024_line1_full_ribozyme`. The two CPEB3 assays stay separate, because
Roberts 2023 uses the ancestral mammalian sequence and Zhang 2020 the human one,
and ProteinGym keeps orthologs apart. Coding assays use their UniProt entry
name. A data directory set up before this column existed needs the current
`reference_sheet_final.csv`, or `performance_fitness` stops with a missing
column error.

## Alignment queries

`ALIGNMENT_QUERY_SEQ` is `RAW_CONSTRUCT_SEQ` with every position that must not
recruit homologs replaced by `N`: T7 leaders, PCR primer constants, Illumina
adapters, cloning sites, vector context and designed linkers. Allowed into a
homology search, those tracts recruit hits on generic experimental identity,
such as other members of the same SELEX library or any adapter-contaminated
database entry, and the resulting alignment describes the experiment rather
than the RNA.

The test is whether a tract can independently recruit sequence through generic
experimental context. Neither whether it folds nor whether it is mutated
settles that: Kobori's randomized communication bases are all those assays
measure and are still designed synthetic sequence, and Janzen's three-prime
constant carries the mapped aminoacylation site and is still 24 nt shared
across the whole N21 library.

The column masks rather than trims, so it is the same length as the construct
and variant positions still map directly. Two masked regions are internal,
`Kobori_2015_ribozyme_j12` 30-33 and `Kobori_2015_ribozyme_p4` 105-108, where
trimming would splice the HDV-like scaffold onto the guanine aptamer and make a
junction present in no real molecule. Coordinates are one-based and inclusive,
as the `mutant` labels are.

The 29 scored ncRNA assays carry a query; 14 of them are masked and 15 are
fully biological and carry the unmasked construct. Rows without an assessment
are empty. Nothing reads the column yet: `rnagym/sequences.py` still builds
alignment queries from `RAW_CONSTRUCT_SEQ`.

## Assay lookup

Use the `DMS_ID` from
[`reference_sheet_final.csv`](reference_sheet_final.csv)
directly, for example `msa/riboseek/by_assay/Kobori_2018_ribozyme.a3m`. The canonical
`sequence_<id>` files are shared by exact sequence, so assays with identical
wild-type sequences resolve to the same alignment.
