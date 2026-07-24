# RNAGym 2D datasets

| Dataset | Contents | Unique key |
| --- | --- | --- |
| `rnagym_2d.parquet` | 947k chemical mapping profiles for 585k sequences, clustered at 40% ID, split into 80/20 train/test. | `seqID` |
| `rnagym_pseudobase.parquet` | 358 sequence/structure pairs with pseudoknots | `pseudobase_ids` |

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
| `cluster_id` | MMseqs2 cluster identifier. |
| `split` | `train` or `test`. |

### `rnagym_pseudobase.parquet` schema

| Column | Description |
| --- | --- |
| `pseudobase_ids` | Source PseudoBase IDs. |
| `sequence` | RNA sequence. |
| `secondary_structure` | Structure in dot-bracket notation. |

## Reproducing the datasets

From the repository root, reconstruct both datasets from `raw_data/` with:

```bash
cd 2d
pixi run collate-2d
```

Chemical mapping sequences are clustered with MMseqs2 at 40% sequence identity
and 80% coverage. Clusters are split 80/20 into train/test with random seed 42.

See [`raw_data/README.md`](raw_data/README.md) for sources and schemas.
