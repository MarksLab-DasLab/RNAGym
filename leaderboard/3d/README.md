# RNAGym 3D Leaderboard

**Metrics:** TM score measures global fold similarity. ΔTM is the prediction
TM score minus the best PDB-chain TM score from before the model's training
cutoff, while ρTM is their Spearman correlation. INF-WC and INF-NWC measure
recovery of Watson-Crick and non-Watson-Crick interactions. INF is one
when both structures lack that interaction class and zero when only one
does. Monomer metrics take the best match across experimental structures
with the same sequence. Monomer sequence scores and individual multimer
target scores are then averaged within each 40% sequence identity cluster,
then across clusters. Missing predictions score zero, and all metrics are
better when higher. Models are ranked by TM score.

<!-- BEGIN GENERATED TABLES -->
### Monomers (n=967 structures)

| Rank | Model | TM | ΔTM | ρTM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | AlphaFold 3 | 0.491 | -0.128 | 0.81 | 0.85 | 0.43 |
| 2 | NuFold | 0.484 | -0.139 | 0.82 | 0.79 | 0.32 |
| 3 | RoseTTAFold2NA | 0.458 | -0.144 | 0.84 | 0.75 | 0.32 |
| 4 | RhoFold+ | 0.457 | -0.168 | 0.81 | 0.53 | 0.09 |
| 5 | trRosettaRNA | 0.414 | -0.209 | 0.67 | 0.70 | 0.14 |

### Multimers (n=455 structures)

| Rank | Model | TM | ΔTM | ρTM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | AlphaFold 3 | 0.403 | -0.096 | 0.78 | 0.89 | 0.67 |
| 2 | RoseTTAFold2NA | 0.190 | -0.294 | 0.29 | 0.49 | 0.37 |
<!-- END GENERATED TABLES -->

Targets were sourced from the [RNA3DB 2026-01-05 full release][1].

[1]: https://github.com/marcellszi/rna3db/releases/tag/2026-01-05-full-release
