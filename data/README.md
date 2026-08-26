# RNAGym data

| File | Contents | Unique key |
| --- | --- | --- |
| `rnagym_sequences.parquet` | Exact sequences, stable identifiers, 40% identity clusters, and folds | `sequence_id` |
| `rnagym_rfams.parquet` | Rfam hits for each registered sequence | `sequence_id` |

To regenerate the Rfam annotations, run from the repository root:

```bash
cd rnagym/s2d
RNAGYM_DATABASE_DIR=/path/to/databases pixi run annotate-rfam
```
