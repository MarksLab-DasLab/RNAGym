# RMDB compiled datasets

This directory is a comprehensive compilation of chemical mapping datasets from
RMDB. Data is stored in Parquet format. Each `.txt` file contains a list of
`rdat` filenames that make up the chemical mapping data in the corresponding
Parquet file.

## Dataset groups

| Dataset | Description |
| --- | --- |
| `RMDB_dataset_<integer>` | RMDB data from the Ribonanza train and test sets. |
| `RMDB_dataset_extra` | Other RMDB data not used in the Ribonanza train and test sets. |

For more granular detail, `RMDB_dataset_extra` is further divided into:

| Dataset | Description |
| --- | --- |
| `extra_clean` | Normal chemical mapping data. |
| `extra_cotrans` | RNA folded cotranscriptionally. |
| `extra_degradation` | Measurements of where RNA is degraded rather than reactivity. |
| `extra_invivo` | Chemical mapping experiments performed *in vivo*. |

## Parquet schema

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
