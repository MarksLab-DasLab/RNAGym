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
### Monomers (n=854 structures)

| Rank | Model | TM | ΔTM | ρTM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | AlphaFold 3 | 0.486 | -0.128 | 0.81 | 0.85 | 0.43 |
| 2 | NuFold | 0.479 | -0.139 | 0.82 | 0.79 | 0.32 |
| 3 | RhoFold+ | 0.458 | -0.162 | 0.81 | 0.54 | 0.10 |
| 4 | RoseTTAFold2NA | 0.454 | -0.142 | 0.84 | 0.75 | 0.32 |
| 5 | trRosettaRNA | 0.415 | -0.202 | 0.68 | 0.71 | 0.13 |

### Multimers (n=404 structures)

| Rank | Model | TM | ΔTM | ρTM | INF-WC | INF-NWC |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | AlphaFold 3 | 0.407 | -0.096 | 0.77 | 0.88 | 0.66 |
| 2 | RoseTTAFold2NA | 0.194 | -0.295 | 0.30 | 0.50 | 0.38 |
<!-- END GENERATED TABLES -->

Targets were sourced from the [RNA3DB 2026-01-05 full release][1].

[1]: https://github.com/marcellszi/rna3db/releases/tag/2026-01-05-full-release
