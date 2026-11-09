# RNAGym clinical data

| File or directory | Contents |
| --- | --- |
| `variants/` | The released benchmark files and their export summaries (below) |
| `staged/` | The files as scoring reads them, from `prepare` |
| `scores/` | One shard directory per file and model, from `score` |
| `merged/` | One parquet per file and model, from `merge` |
| `evaluation/` | Per-class AUROC tables, from `evaluate` |

## Released files

`variants/` holds what `prepare` reads. It checks each CSV against the sha256 in
its export summary.

| File | Variants | Sequence |
| --- | --- | --- |
| `premrna_10101_snv.csv` | ClinVar SNVs | 10,101 nt of the gene's strand, centred on the variant |
| `mature_mrna_snv.csv` | ClinVar SNVs in exons | The spliced transcript |
| `premrna_10101_indel.csv` | ClinVar insertions and deletions | As the pre-mRNA SNVs |
| `mature_mrna_indel.csv` | ClinVar insertions and deletions in exons | The spliced transcript |
| `mt_rna.csv` | MITOMAP variants in mitochondrial RNAs | The processed mRNA, rRNA or tRNA |
| `mt_trna_cca.csv` | MITOMAP tRNA variants | The processed tRNA with its CCA tail |
| `export_summary.json`, `mito_export_summary.json` | | Row counts and sha256 of every file |

`merged/` also comes with the release: the published scores of every model, so
the leaderboard can be rebuilt without a GPU. The pre-mRNA scores were made
from the files before a later correction of `consequence_groups`, which changed
no sequence or label, so `merge` cannot remake them from `staged/`; they
evaluate as they are, joined on `row_id`.
