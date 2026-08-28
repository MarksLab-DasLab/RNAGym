# RNAGym data

| File | Contents | Unique key |
| --- | --- | --- |
| `rnagym_sequences.parquet` | Exact sequences, stable identifiers, 40% identity clusters, and folds | `sequence_id` |
| `rnagym_rfams.parquet` | Rfam hits for each registered sequence | `sequence_id` |
| `3d/msa/` | Riboseek alignments used for 3D prediction | `sequence_id` |
| `fitness/msa/` | Riboseek alignments used for fitness scoring | `sequence_id` |

To browse the Parquet datasets as interactive tables:

```bash
cd data
pixi run browse
```

To regenerate the Rfam annotations, run from the repository root:

```bash
cd rnagym/s2d
RNAGYM_DATABASE_DIR=/path/to/databases pixi run annotate-rfam
```
