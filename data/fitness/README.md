# RNAGym fitness data

| File or directory | Contents |
| --- | --- |
| `msa/` | Riboseek alignments for unique ncRNA fitness sequences |
| `msa/by_assay/` | Alignment symlinks named by the paper's `DMS_ID` assay key |

## Assay lookup

Use the `DMS_ID` from
[`fitness/reference_sheet_final.csv`](../../fitness/reference_sheet_final.csv)
directly, for example `msa/by_assay/Kobori_2018_ribozyme.a3m`. The canonical
`sequence_<id>` files are shared by exact sequence, so assays with identical
wild-type sequences resolve to the same alignment.
