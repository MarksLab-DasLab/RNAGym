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
| 1 | Protenix-v1 | -0.109 | 0.510 | 0.87 | 0.47 |
| 2 | OpenFold3 | -0.117 | 0.502 | 0.86 | 0.45 |
| 3 | RoseTTAFold3 | -0.122 | 0.497 | 0.85 | 0.43 |
| 4 | AlphaFold 3 | -0.128 | 0.491 | 0.85 | 0.43 |
| 5 | NuFold | -0.139 | 0.484 | 0.79 | 0.32 |
| 6 | RoseTTAFold2NA | -0.144 | 0.458 | 0.75 | 0.32 |
| 7 | RhoFold+ | -0.168 | 0.457 | 0.53 | 0.09 |
| 8 | Boltz-2 | -0.196 | 0.491 | 0.85 | 0.40 |
| 9 | trRosettaRNA | -0.209 | 0.414 | 0.70 | 0.14 |

### Multimers (n=455 structures)

| Rank | Model | ΔTM | TM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | AlphaFold 3 | -0.096 | 0.403 | 0.89 | 0.67 |
| 2 | Protenix-v1 | -0.098 | 0.400 | 0.89 | 0.66 |
| 3 | RoseTTAFold3 | -0.115 | 0.383 | 0.88 | 0.64 |
| 4 | OpenFold3 | -0.119 | 0.380 | 0.88 | 0.69 |
| 5 | Boltz-2 | -0.193 | 0.395 | 0.87 | 0.66 |
| 6 | RoseTTAFold2NA | -0.294 | 0.190 | 0.49 | 0.37 |
<!-- END GENERATED TABLES -->

**Metrics** (higher is better for all)

- **ΔTM:** prediction TM minus the TM of the closest pre-cutoff PDB chain.
- **TM:** global fold similarity to the experimental structure.
- **INF-WC, INF-NWC:** recovery of Watson-Crick and non-Watson-Crick
  interactions. INF is 1 when neither structure has that interaction type and 0
  when only one does.

**Scoring:** Monomers keep the best match across experimental structures of the
same sequence, and each multimer chain counts separately. Scores are averaged
within 40% sequence-identity clusters, then across clusters. Failed predictions
score 0. Targets come from the [RNA3DB 2026-01-05 full release][1].

[1]: https://github.com/marcellszi/rna3db/releases/tag/2026-01-05-full-release
