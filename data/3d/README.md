# RNAGym 3D data

| File or directory | Contents |
| --- | --- |
| `rnagym_3d.parquet` | 967 monomer and 455 multimer RNA chain targets |
| `rnagym_3d_scores.parquet` | Released per-target baseline scores |
| `curation/` | RNA3DB chain annotations used for filtering |
| `curation/.cache/` | Downloaded PDB assemblies and derived per-chain files |
| `predictions/` | Predicted structures grouped by model |
| `msa/riboseek/` | Riboseek input alignments for unique sequences |
| `msa/mmseqs/` | Conventional nucleotide MMseqs2 comparison alignments |
| `usalign/` | Precomputed test-to-reference US-align outputs |

The dataset curation pipeline is documented in
[`rnagym/s3d/README.md`](../../rnagym/s3d/README.md). From that directory,
`pixi run leaderboard` reproduces the [3D leaderboard](../../leaderboard/3d/).
