# RNAGym 2D Leaderboard

**Chemical mapping:** Cluster-macro Spearman compares finite reactivities (most
NaNs are unmeasured construct regions such as barcodes) with predicted unpaired
probabilities (`1 − Σj Pij`). For neural models, highly confident residues are
clipped to 1 following
[Arnie's official RibonanzaNet inference](https://github.com/WaymentSteeleLab/arnie/blob/660de8139bd2198bbe115adadd5bc5f12183f9f4/src/arnie/pk_predictors.py#L111-L116)
when summed pair probabilities exceed one at any given residue. DMS is scored
over A/C, CMCT over G/U, and other modalities over all bases. The headline score
averages 1M7, 2A3, DMS, and NMIA.
BzCN (low replicate agreement), CMCT (only 11 replicate-bearing clusters), and
degradation assays are reported only in [`leaderboard.csv`](leaderboard.csv).
Constant predictions receive zero.

**Discrete structures:** Cluster-macro base-pair F1 uses each model's
official decoders. PDB pairs touching unresolved residues are excluded.
For repeated structures of the same sequence, the best match is kept before
cluster averaging. Unsupported sequences receive zero. Each model uses
one decoder across PDB, bpRNA-1m, and eFold Challenging, the one with the best
mean score over the three; PseudoBase reports each model's best decoder so
pseudoknot-capable decoders can be used there. eFold
Challenging tests generalization to long noncoding and viral RNAs beyond the
short RNAs and/or Rfam-derived families prevalent in standard datasets. Macro
is the unweighted mean of chemical mapping and the four discrete structure
benchmarks.

## Leaderboard

<!-- BEGIN GENERATED TABLE -->
| Rank | Model | Chemical mapping (n=584k) | PseudoBase (n=358) | PDB (n=1,088) | bpRNA-1m (n=56k) | eFold Challenging (n=45) | Macro |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RibonanzaNet*†‡§ | 0.3899 | 0.7684 | 0.8480 | 0.5525 | 0.6881 | 0.6494 |
| 2 | Vienna | 0.4270 | 0.7123 | 0.7764 | 0.5176 | 0.7276 | 0.6322 |
| 3 | EternaFold* | 0.4686 | 0.6175 | 0.7845 | 0.5384 | 0.7442 | 0.6306 |
| 4 | RNAstructure | 0.4219 | 0.6323 | 0.7772 | 0.5118 | 0.7345 | 0.6155 |
| 5 | CONTRAfold | 0.4375 | 0.6123 | 0.7780 | 0.5353 | 0.7120 | 0.6150 |
| 6 | UFold†‡§ | 0.3012 | 0.6565 | 0.8082 | 0.6663 | 0.5713 | 0.6007 |
| 7 | MXFold2†‡§ | 0.3615 | 0.5977 | 0.8020 | 0.5249 | 0.6667 | 0.5906 |
| 8 | RNA-FM†‡§ | 0.2866 | 0.6689 | 0.8040 | 0.5920 | 0.5052 | 0.5713 |
| 9 | RiNALMo†‡§ | 0.2586 | 0.4639 | 0.6860 | 0.7670 | 0.2195 | 0.4790 |
<!-- END GENERATED TABLE -->

\*, †, and ‡ indicate training data from the chemical mapping, PseudoBase, and
PDB source collections. § indicates training or pretraining on bpRNA-1m,
RNAcentral, or Rfam because these collections overlap heavily.
