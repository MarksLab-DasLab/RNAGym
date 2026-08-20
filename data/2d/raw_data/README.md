# Compiled secondary-structure source data

Raw data from [RMDB][rmdb], [PseudoBase++2.0][pb] (see [Taufer et al.
(2009)][taufer]), [bpRNA-1m][bprna-download], and the [eFold Challenging
dataset][efold-dryad].

## Dataset groups

| Dataset | Description |
| --- | --- |
| `RMDB_dataset_<integer>` | RMDB data from the Ribonanza train and test sets. |
| `RMDB_dataset_extra` | Other RMDB data not used in the Ribonanza train and test sets. |
| `pseudobase.csv` | Entries from PseudoBase (downloaded 2026/07/20) |
| `bprna_1m_dbn.zip` | Official bpRNA-1m v1.0 dot-bracket archive. |
| `lncRNA_nonFiltered.json` | eFold Challenging long noncoding RNA structures. |
| `viral_fragments.json` | eFold Challenging viral-fragment structures. |

For more granular detail, `RMDB_dataset_extra` is further divided into:

| Dataset | Description |
| --- | --- |
| `extra_clean` | Normal chemical mapping data. |
| `extra_cotrans` | RNA folded cotranscriptionally. |
| `extra_degradation` | Measurements of where RNA is degraded rather than reactivity. |
| `extra_invivo` | Chemical mapping experiments performed *in vivo*. |

Collation keeps canonical bpRNA-1m sequences and collapses exact duplicates.
The eFold base-pair lists are converted to extended dot-bracket notation.

[rmdb]: https://rmdb.stanford.edu/
[pb]: https://rnavlab.utep.edu/database
[taufer]: https://doi.org/10.1093/nar/gkn806
[bprna-download]: https://bprna.cgrb.oregonstate.edu/download.php
[efold-dryad]: https://doi.org/10.5061/dryad.79cnp5j95
