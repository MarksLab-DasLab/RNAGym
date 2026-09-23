# RNAGym fitness data

| File or directory | Contents |
| --- | --- |
| `assays/` | Processed DMS assay tables |
| `merged/` | Assays joined to model predictions |
| `model_predictions/` | One prediction directory per model |
| `msa/riboseek/` | Riboseek alignments for unique ncRNA fitness sequences |
| `msa/mmseqs/` | Conventional nucleotide MMseqs2 comparison alignments |
| `msa/riboseek/by_assay/`, `msa/mmseqs/by_assay/` | Alignment symlinks named by the paper's `DMS_ID` assay key |
| `reference_sheet_final.csv` | Assay metadata and wild-type constructs |
| `reports/` | Supplemental fitness analysis tables |

## Keeping the directory in step with the reference sheet

`assays/` holds one CSV per row of `reference_sheet_final.csv`, and
`merge_scoring_files` rejects any processed assay the sheet does not list. When
an assay is retired from the sheet, delete its `assays/` table and its
`model_predictions/*/` files before the next merge. Two assays were retired
after v0.2, `Beck_2022_ribozyme` and `Kobori_2016_osa_ribozyme`, both duplicates
of assays that remain.

## Assay lookup

Use the `DMS_ID` from
[`reference_sheet_final.csv`](reference_sheet_final.csv)
directly, for example `msa/riboseek/by_assay/Kobori_2018_ribozyme.a3m`. The canonical
`sequence_<id>` files are shared by exact sequence, so assays with identical
wild-type sequences resolve to the same alignment.
