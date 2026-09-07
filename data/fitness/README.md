# RNAGym fitness data

| File or directory | Contents |
| --- | --- |
| `assays/` | Processed DMS assay tables |
| `merged/` | Assays joined to model predictions |
| `model_predictions/` | One prediction directory per model |
| `msa/` | Riboseek alignments for unique ncRNA fitness sequences |
| `msa/by_assay/` | Alignment symlinks named by the paper's `DMS_ID` assay key |
| `reference_sheet_final.csv` | Assay metadata and wild-type constructs |
| `reports/` | Supplemental fitness analysis tables |

## Assay lookup

<!--TODO(MCA/TLC): Address the 2 duplicates -->
Use the `DMS_ID` from
[`reference_sheet_final.csv`](reference_sheet_final.csv)
directly, for example `msa/by_assay/Kobori_2018_ribozyme.a3m`. The canonical
`sequence_<id>` files are shared by exact sequence, so assays with identical
wild-type sequences resolve to the same alignment.
