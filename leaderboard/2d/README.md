# RNAGym 2D Leaderboard

**Chemical mapping:** Cluster-macro Spearman compares finite reactivities (most
NaNs are unmeasured construct regions such as barcodes) with predicted unpaired
probabilities (`1 − Σj Pij`). For neural models, highly confident residues are
clipped to 1 following
[Arnie's official RibonanzaNet inference](https://github.com/WaymentSteeleLab/arnie/blob/660de8139bd2198bbe115adadd5bc5f12183f9f4/src/arnie/pk_predictors.py#L111-L116)
when summed pair probabilities exceed one at any given residue. DMS is scored
over A/C, CMCT over G/U, and other modalities over all bases. Spearman avoids
directly comparing reactivity magnitudes with probabilities or choosing a
classification threshold. The headline score averages 1M7, 2A3, DMS, and NMIA.
BzCN (low replicate agreement), CMCT (only 11 replicate-bearing clusters), and
degradation assays are reported only in [`leaderboard.csv`](leaderboard.csv).
Constant predictions receive zero.

**Discrete structures:** Cluster-macro base-pair F1 uses each model's
official decoders. PDB pairs touching unresolved residues are excluded.
For repeated structures of the same sequence, the best match is kept before
cluster averaging. Unsupported sequences receive zero. PseudoBase, PDB,
bpRNA-1m, and eFold Challenging report each model's best decoder. eFold
Challenging tests generalization to long noncoding and viral RNAs beyond the
short RNAs and/or Rfam-derived families prevalent in standard datasets. Macro
is the unweighted mean of chemical mapping and the four discrete structure
benchmarks.

## Leaderboard

<!-- BEGIN GENERATED TABLE -->
| Rank | Model | Chemical mapping (n=584k) | PseudoBase (n=358) | PDB (n=1,025) | bpRNA-1m (n=56k) | eFold Challenging (n=45) | Macro |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RibonanzaNet*‡§ | 0.4028 | 0.7673 | 0.8430 | 0.5672 | 0.6992 | 0.6559 |
| 2 | EternaFold* | 0.4880 | 0.6142 | 0.7720 | 0.5423 | 0.7523 | 0.6338 |
| 3 | Vienna | 0.4479 | 0.7051 | 0.7589 | 0.5171 | 0.7358 | 0.6330 |
| 4 | CONTRAfold | 0.4586 | 0.6054 | 0.7630 | 0.5387 | 0.7220 | 0.6175 |
| 5 | RNAstructure | 0.4427 | 0.6250 | 0.7589 | 0.5146 | 0.7454 | 0.6173 |
| 6 | UFold‡§ | 0.3212 | 0.6517 | 0.8063 | 0.6844 | 0.5761 | 0.6079 |
| 7 | MXFold2§ | 0.3808 | 0.5939 | 0.7990 | 0.5484 | 0.6805 | 0.6005 |
| 8 | RNA-FM‡§ | 0.3053 | 0.6615 | 0.8011 | 0.6093 | 0.5164 | 0.5787 |
| 9 | RiNALMo‡§ | 0.2810 | 0.4564 | 0.6949 | 0.7961 | 0.2199 | 0.4897 |
<!-- END GENERATED TABLE -->

\*, †, and ‡ indicate training data from the chemical mapping, PseudoBase, and
PDB source collections. § indicates training or pretraining on bpRNA-1m,
RNAcentral, or Rfam because these collections overlap heavily.
