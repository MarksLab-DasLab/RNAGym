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
| 1 | Protenix-v1 | -0.110 | 0.490 | 0.86 | 0.44 |
| 2 | OpenFold3 | -0.121 | 0.480 | 0.85 | 0.42 |
| 3 | RoseTTAFold3 | -0.128 | 0.472 | 0.84 | 0.41 |
| 4 | AlphaFold 3 | -0.135 | 0.466 | 0.84 | 0.40 |
| 5 | NuFold | -0.146 | 0.460 | 0.78 | 0.29 |
| 6 | RoseTTAFold2NA | -0.155 | 0.430 | 0.74 | 0.29 |
| 7 | RhoFold+ | -0.180 | 0.428 | 0.53 | 0.09 |
| 8 | trRosettaRNA | -0.210 | 0.395 | 0.70 | 0.12 |
| 9 | Boltz-2 | -0.212 | 0.464 | 0.83 | 0.37 |

### Multimers (n=455 structures)

| Rank | Model | ΔTM | TM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | AlphaFold 3 | -0.098 | 0.393 | 0.88 | 0.67 |
| 2 | Protenix-v1 | -0.100 | 0.390 | 0.88 | 0.66 |
| 3 | RoseTTAFold3 | -0.121 | 0.370 | 0.87 | 0.65 |
| 4 | OpenFold3 | -0.123 | 0.367 | 0.88 | 0.68 |
| 5 | Boltz-2 | -0.191 | 0.383 | 0.87 | 0.65 |
| 6 | RoseTTAFold2NA | -0.284 | 0.191 | 0.50 | 0.38 |
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
