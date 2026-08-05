# RNAGym 2D datasets

| Dataset | Contents | Unique key |
| --- | --- | --- |
| `rnagym_mapping.parquet` | 970k chemical mapping profiles for 585k sequences | `uid` |
| `rnagym_2d.parquet` | 358 PseudoBase and 2,062 PDB structures | `uid` |
| `rnagym_sequences.parquet` | Global sequence registry, clusters, and folds | `sequence_id` |
| `rnagym_rfams.parquet` | Rfam hits for each registered sequence | `sequence_id` |

### `rnagym_mapping.parquet` schema

| Column | Description |
| --- | --- |
| `uid` | Unique representative chemical mapping profile identifier. |
| `sequence_id` | Identifier shared by profiles with the same sequence. |
| `sequence` | RNA sequence. |
| `modifier` | Chemical modifier used to modify the RNA. |
| `SNR` | Signal-to-noise ratio. |
| `reads` | Total number of Illumina sequencing reads assigned to the profile; not available for datasets labeled `extra`. |
| `temperature` | Experimental temperature. |
| `chemical` | Chemicals used in the experiment, such as buffer or salt. |
| `reverse_transcriptase` | Reverse transcriptase used to read out the chemical modification. |
| `note` | Additional notes about the data. |
| `reactivity` | Per-nucleotide reactivity values. |
| `reactivity_error` | RMDB-provided values correlated with measurement error. Their definition varies and is often undocumented across entries, so they are not considered when scoring. |
| `replicates` | Additional measurements storing `uid`, `reactivity`, `reactivity_error`, `SNR`, and `reads`. |

Replicates match the RMDB series, sequence, modifier, temperature, chemicals,
reverse transcriptase, note, and context. Some source rows incorrectly assign
different UIDs to identical measurements. These duplicates are collapsed, and
the highest-SNR measurement is the representative.

### `rnagym_2d.parquet` schema

| Column | Description |
| --- | --- |
| `uid` | Unique structure identifier formatted as `<source>:<source_id>`. |
| `sequence_id` | Identifier shared by records with the same sequence. |
| `sequence` | RNA sequence. |
| `secondary_structure` | Structure in dot-bracket notation. |
| `resolved` | Positions resolved in the source structure and included in scoring. |

PDB entries are canonical RNA monomers selected from `3d/annotated_chains.csv`
using [`Config3D`](../../../config.py). Structures contain cis
Watson-Crick/Watson-Crick pairs assigned by the pinned RNA-Puzzles MC-Annotate.
Contacts involving residues with multiple such partners are excluded because
dot-bracket cannot represent them.

### `rnagym_sequences.parquet` schema

| Column | Description |
| --- | --- |
| `sequence_id` | Stable identifier for an exact RNA sequence. |
| `sequence` | RNA sequence. |
| `cluster_rep` | `sequence_id` of the global MMseqs2 cluster representative. |
| `fold` | Fold assignment. |

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

Clustering and fold assignment use [`Config2D`](../../../config.py).

See [`raw_data/README.md`](raw_data/README.md) for sources and schemas.

### Optionally annotate each unique sequence with Rfam

To process the dataset with Rfam/Infernal and (re)generate
`rnagym_rfams.parquet`:

```bash
pixi run annotate-rfam
```
