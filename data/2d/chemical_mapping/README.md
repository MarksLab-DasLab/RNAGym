# RNAGym 2D datasets

| Dataset | Contents | Unique key |
| --- | --- | --- |
| `rnagym_map.parquet` | 947k chemical mapping profiles for 585k sequences | `uid` |
| `rnagym_pb.parquet` | 358 sequence/structure pairs with pseudoknots | `pseudobase_ids` |
| `rnagym_sequences.parquet` | Global sequence registry, clusters, and folds | `sequence_id` |
| `rnagym_rfams.parquet` | Rfam 15.1 hits for each registered sequence | `sequence_id` |

### `rnagym_map.parquet` schema

| Column | Description |
| --- | --- |
| `uid` | Unique chemical mapping profile identifier. |
| `sequence_id` | Identifier shared by profiles with the same sequence. |
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
### `rnagym_pb.parquet` schema

| Column | Description |
| --- | --- |
| `pseudobase_ids` | Source PseudoBase IDs. |
| `sequence_id` | Identifier shared by records with the same sequence. |
| `sequence` | RNA sequence. |
| `secondary_structure` | Structure in dot-bracket notation. |

### `rnagym_sequences.parquet` schema

| Column | Description |
| --- | --- |
| `sequence_id` | Stable identifier for an exact RNA sequence. |
| `sequence` | RNA sequence. |
| `cluster_rep` | `sequence_id` of the global MMseqs2 cluster representative. |
| `fold` | Fold assignment from 0–4. |

### `rnagym_rfams.parquet` schema

| Column | Description |
| --- | --- |
| `sequence_id` | Identifier from `rnagym_sequences.parquet`. |
| `rfam_hits` | Rfam accession, name, clan, E-value, score, coordinates, strand, truncation, and overlap for each hit. |

## Reproducing the datasets

From the repository root, reconstruct the datasets from `raw_data/` with:

```bash
cd 2d
pixi run collate-2d
```

All sequences are clustered together with MMseqs2 at 40% sequence identity and
80% coverage. Clusters are randomly assigned to 5 distinct folds to support
community train/test splits and cross-validation.

See [`raw_data/README.md`](raw_data/README.md) for sources and schemas.

### Optionally annotate each unique sequence with Rfam

To process the dataset with Rfam/Infernal and (re)generate
`rnagym_rfams.parquet`:

```bash
pixi run annotate-rfam
```
