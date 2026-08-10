# RNAGym 3D Leaderboard

**Metrics:** TM score measures global fold similarity. ΔTM is the prediction TM
score minus the best training-template TM score, while ρTM is their Spearman
correlation. INF-WC and INF-NWC measure recovery of Watson-Crick and
non-Watson-Crick interactions. Undefined INF values are excluded. All metrics
are averaged across test chains, with missing predictions scored as zero, and
are better when higher. Models are ranked by TM score.

<!-- BEGIN GENERATED TABLES -->
### Monomers (n=88)

| Rank | Model | n | TM | ΔTM | ρTM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | NuFold | 88 | 0.393 | -0.151 | 0.74 | 0.78 | 0.19 |
| 2 | AlphaFold 3 | 88 | 0.386 | -0.167 | 0.71 | 0.83 | 0.26 |
| 3 | trRosettaRNA | 88 | 0.374 | -0.170 | 0.50 | 0.73 | 0.12 |
| 4 | RoseTTAFold2NA | 88 | 0.365 | -0.154 | 0.67 | 0.74 | 0.21 |
| 5 | RhoFold+ | 88 | 0.363 | -0.181 | 0.67 | 0.49 | 0.08 |

### Multimers (n=127)

| Rank | Model | n | TM | ΔTM | ρTM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | AlphaFold 3 | 127 | 0.381 | -0.130 | 0.55 | 0.86 | 0.37 |
| 2 | RoseTTAFold2NA | 127 | 0.168 | -0.294 | 0.23 | 0.18 | 0.00 |
<!-- END GENERATED TABLES -->
