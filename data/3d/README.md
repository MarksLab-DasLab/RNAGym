# RNAGym 3D data

| File or directory | Contents |
| --- | --- |
| `rnagym_3d.parquet` | 88 monomer and 127 multimer evaluation targets |
| `rnagym_3d_scores.parquet` | Released per-target baseline scores |
| `curation/` | Candidate PDB entries and annotated RNA chains |
| `predictions/` | Predicted structures grouped by model |
| `structures/` | Experimental and predicted PDB structures |
| `msa/` | Input multiple sequence alignments |
| `usalign/` | Precomputed test-to-reference US-align outputs |

The dataset curation pipeline is documented in
[`rnagym/s3d/README.md`](../../rnagym/s3d/README.md). From that directory,
`pixi run leaderboard` reproduces the [3D leaderboard](../../leaderboard/3d/).
