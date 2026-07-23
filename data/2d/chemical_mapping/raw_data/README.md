# Compiled source data (RMDB + PseudoBase++2.0)

Raw data from [RMDB][rmdb] and [PseudoBase++2.0][pb] (see [Taufer et al.
(2009)][taufer]).

## Dataset groups

| Dataset | Description |
| --- | --- |
| `RMDB_dataset_<integer>` | RMDB data from the Ribonanza train and test sets. |
| `RMDB_dataset_extra` | Other RMDB data not used in the Ribonanza train and test sets. |
| `pseudobase.csv` | Entries from PseudoBase (downloaded 2026/07/20) |

For more granular detail, `RMDB_dataset_extra` is further divided into:

| Dataset | Description |
| --- | --- |
| `extra_clean` | Normal chemical mapping data. |
| `extra_cotrans` | RNA folded cotranscriptionally. |
| `extra_degradation` | Measurements of where RNA is degraded rather than reactivity. |
| `extra_invivo` | Chemical mapping experiments performed *in vivo*. |

[rmdb]: https://rmdb.stanford.edu/
[pb]: https://rnavlab.utep.edu/database
[taufer]: https://doi.org/10.1093/nar/gkn806
