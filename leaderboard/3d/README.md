# RNAGym 3D Leaderboard

Models are ranked by ΔTM, which measures how well each model memorizes its
training set and generalizes beyond it. ΔTM is the prediction's TM score minus
that of the closest PDB structure released before the model's training cutoff.
Zero matches recalling the best training structure, and positive values mean
the model generalizes beyond its training data. Metric details are below the
tables.

<!-- BEGIN GENERATED TABLES -->
### Monomers (n=967 structures)

| Rank | Model | ΔTM | TM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | Protenix-v1 | -0.108 | 0.492 | 0.86 | 0.45 |
| 2 | OpenFold3 | -0.117 | 0.482 | 0.85 | 0.43 |
| 3 | RoseTTAFold3 | -0.127 | 0.472 | 0.84 | 0.42 |
| 4 | AlphaFold 3 | -0.136 | 0.464 | 0.84 | 0.41 |
| 5 | NuFold | -0.144 | 0.460 | 0.78 | 0.30 |
| 6 | RoseTTAFold2NA | -0.151 | 0.431 | 0.73 | 0.30 |
| 7 | RhoFold+ | -0.172 | 0.436 | 0.53 | 0.08 |
| 8 | trRosettaRNA | -0.206 | 0.398 | 0.70 | 0.12 |
| 9 | Boltz-2 | -0.209 | 0.467 | 0.83 | 0.38 |

### Multimers (n=455 structures)

| Rank | Model | ΔTM | TM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | AlphaFold 3 | -0.097 | 0.394 | 0.88 | 0.66 |
| 2 | Protenix-v1 | -0.100 | 0.391 | 0.88 | 0.66 |
| 3 | RoseTTAFold3 | -0.120 | 0.370 | 0.87 | 0.63 |
| 4 | OpenFold3 | -0.122 | 0.369 | 0.87 | 0.68 |
| 5 | Boltz-2 | -0.190 | 0.384 | 0.87 | 0.64 |
| 6 | RoseTTAFold2NA | -0.284 | 0.193 | 0.49 | 0.38 |
<!-- END GENERATED TABLES -->

**Metrics** (higher is better for all)

- **ΔTM:** prediction TM minus the TM of the closest pre-cutoff PDB chain.
- **TM:** global fold similarity to the experimental structure.
- **INF-WC, INF-NWC:** recovery of Watson-Crick and non-Watson-Crick
  interactions. INF is 1 when neither structure has that interaction type and 0
  when only one does.

**Scoring:** Monomers keep the best match across experimental structures of the
same sequence, and each multimer chain counts separately. Scores are averaged
within 50% sequence-identity clusters, then across clusters. Failed predictions
score 0. Targets come from the [RNA3DB 2026-01-05 full release][1].

[1]: https://github.com/marcellszi/rna3db/releases/tag/2026-01-05-full-release
