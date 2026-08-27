# RNAGym 2D data

## Datasets

| Dataset | Contents | Unique key |
| --- | --- | --- |
| `rnagym_mapping.parquet` | 970k chemical mapping profiles for 585k sequences | `uid` |
| `rnagym_2d.parquet` | 57,120 bpRNA-1m, 45 eFold Challenging, 358 PseudoBase, and 2,566 PDB structures | `uid` |

## Schemas

### `rnagym_mapping.parquet`

| Column | Description |
| --- | --- |
| `uid` | Unique representative chemical mapping profile identifier. |
| `sequence_id` | Identifier shared by profiles with the same sequence. |
| `sequence` | RNA sequence. |
| `modifier` | Chemical modifier used to modify the RNA. |
| `SNR` | Signal-to-noise ratio. |
| `reads` | Total number of Illumina sequencing reads assigned to the profile. Not available for datasets labeled `extra`. |
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

### `rnagym_2d.parquet`

| Column | Description |
| --- | --- |
| `uid` | Unique structure identifier formatted as `<source>:<source_id>`. |
| `sequence_id` | Identifier shared by records with the same sequence. |
| `sequence` | RNA sequence. |
| `secondary_structure` | Structure in dot-bracket notation. |
| `resolved` | Positions resolved in the source structure and included in scoring. |

PDB entries are canonical RNA monomers selected from
`data/3d/curation/annotated_chains.parquet` using
[`Config3D`](../../rnagym/config.py). Structures contain cis
Watson-Crick/Watson-Crick pairs assigned by the pinned RNA-Puzzles MC-Annotate.
Contacts involving residues with multiple such partners are excluded because
dot-bracket cannot represent them. All residues in these conflicting contact
components are marked unresolved and excluded from scoring.

## Reproducing the datasets

Configure [`Config2D`](../../rnagym/config.py), then from the repository root
reconstruct the datasets with:

```bash
cd rnagym/s2d
pixi run collate-2d
```
