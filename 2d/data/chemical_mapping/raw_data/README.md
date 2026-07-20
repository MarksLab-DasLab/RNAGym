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

### `rnagym_2d.parquet` schema

| Column | Description |
| --- | --- |
| `seqID` | Unique identifier for the data. |
| `sequence` | RNA sequence. |
| `modifier` | Chemical modifier used to modify the RNA. |
| `SNR` | Signal-to-noise ratio. |
| `reads` | Total number of Illumina sequencing reads assigned to the profile; not available for datasets labeled `extra`. |
| `temperature` | Experimental temperature. |
| `chemical` | Chemicals used in the experiment, such as buffer or salt. |
| `reverse_transcriptase` | Reverse transcriptase used to read out the chemical modification. |
| `note` | Additional notes about the data. |
| `reactivity` | Sequence reactivity stored as a string, for example `[0.01,0.10,0.90,...]`. |
| `reactivity_error` | Reactivity error stored in the same string format as `reactivity`. |
| `split` | `train` or `test`. |

### `rnagym_pseudobase.parquet` schema

| Column | Description |
| --- | --- |
| `pseudobase_ids` | Source PseudoBase IDs. |
| `sequence` | RNA sequence. |
| `secondary_structure` | Structure in dot-bracket notation. |

[rmdb]: https://rmdb.stanford.edu/
[pb]: https://rnavlab.utep.edu/database
[taufer]: https://doi.org/10.1093/nar/gkn806
