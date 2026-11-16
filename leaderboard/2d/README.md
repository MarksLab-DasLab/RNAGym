# RNAGym 2D Leaderboard

**Chemical mapping:** Cluster-macro Spearman compares finite reactivities (most
NaNs are unmeasured construct regions such as barcodes) with predicted unpaired
probabilities (`1 − Σj Pij`). For neural models, highly confident residues are
clipped to 1 following
[Arnie's official RibonanzaNet inference](https://github.com/WaymentSteeleLab/arnie/blob/660de8139bd2198bbe115adadd5bc5f12183f9f4/src/arnie/pk_predictors.py#L111-L116)
when summed pair probabilities exceed one at any given residue. DMS is scored
over A/C, CMCT over G/U, and other modalities over all bases. In RMDB and
Ribonanza, the headline score averages 1M7, 2A3, DMS, and NMIA; BzCN (low
replicate agreement), CMCT (only 11 replicate-bearing clusters), and degradation
assays are reported in [`leaderboard.csv`](leaderboard.csv). **eFold viral**
scores raw experimental DMS (25 profiles) and SHAPE (15 profiles) reactivities
across 40 viral RNA fragments without thermodynamic structure modeling, with
per-reagent breakdowns (`efold_dms`, `efold_shape`) in
[`leaderboard.csv`](leaderboard.csv). Constant predictions receive zero.

**Discrete structures:** Cluster-macro base-pair F1 uses each model's
official decoders. PDB pairs touching unresolved residues are excluded.
For repeated structures of the same sequence, the best match is kept before
cluster averaging. Unsupported sequences receive zero. Each model uses
one decoder across PDB and bpRNA-1m, the one with the best mean score over the
two; PseudoBase reports each model's best decoder so pseudoknot-capable decoders
can be used there. Macro is the unweighted mean across the five benchmarks
(chemical mapping, eFold viral, PseudoBase, PDB, and bpRNA-1m).

## Leaderboard

<!-- BEGIN GENERATED TABLE -->
| Rank | Model | Chemical mapping (n=584k) | eFold viral (n=40) | PseudoBase (n=358) | PDB (n=1,088) | bpRNA-1m (n=56k) | Macro |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RibonanzaNet*¶†‡§ | 0.3899 | 0.5265 | 0.7684 | 0.8480 | 0.5525 | 0.6171 |
| 2 | EternaFold* | 0.4686 | 0.6068 | 0.6175 | 0.7845 | 0.5384 | 0.6032 |
| 3 | Vienna | 0.4270 | 0.5776 | 0.7123 | 0.7764 | 0.5176 | 0.6022 |
| 4 | CONTRAfold | 0.4375 | 0.5956 | 0.6123 | 0.7780 | 0.5353 | 0.5917 |
| 5 | RNAstructure | 0.4219 | 0.5685 | 0.6323 | 0.7772 | 0.5118 | 0.5823 |
| 6 | UFold†‡§ | 0.3012 | 0.3969 | 0.6565 | 0.8082 | 0.6663 | 0.5658 |
| 7 | MXFold2†‡§ | 0.3615 | 0.4869 | 0.5977 | 0.8020 | 0.5249 | 0.5546 |
| 8 | RNA-FM†‡§ | 0.2866 | 0.3854 | 0.6689 | 0.8040 | 0.5920 | 0.5474 |
| 9 | RiNALMo†‡§ | 0.2586 | 0.2473 | 0.4639 | 0.6860 | 0.7670 | 0.4846 |
<!-- END GENERATED TABLE -->

\*, ¶, †, and ‡ indicate training data overlapping the chemical mapping, eFold
viral (short Ribonanza viral tiles), PseudoBase, and PDB collections. §
indicates training or pretraining on bpRNA-1m, RNAcentral, or Rfam because these
collections overlap heavily.
