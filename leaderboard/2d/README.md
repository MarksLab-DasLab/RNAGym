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
| 1 | RibonanzaNet*†‡§ | 0.4028 | 0.7680 | 0.8437 | 0.5675 | 0.6971 | 0.6558 |
| 2 | EternaFold* | 0.4882 | 0.6148 | 0.7767 | 0.5427 | 0.7523 | 0.6349 |
| 3 | Vienna | 0.4480 | 0.7058 | 0.7637 | 0.5173 | 0.7358 | 0.6341 |
| 4 | CONTRAfold | 0.4587 | 0.6059 | 0.7679 | 0.5390 | 0.7220 | 0.6187 |
| 5 | RNAstructure | 0.4428 | 0.6252 | 0.7643 | 0.5147 | 0.7451 | 0.6184 |
| 6 | UFold†‡§ | 0.3212 | 0.6519 | 0.8083 | 0.6845 | 0.5761 | 0.6084 |
| 7 | MXFold2†‡§ | 0.3809 | 0.5942 | 0.8026 | 0.5484 | 0.6805 | 0.6013 |
| 8 | RNA-FM†‡§ | 0.3052 | 0.6623 | 0.8031 | 0.6099 | 0.5128 | 0.5787 |
| 9 | RiNALMo†‡§ | 0.2812 | 0.4579 | 0.6931 | 0.7964 | 0.2199 | 0.4897 |
<!-- END GENERATED TABLE -->

\*, †, and ‡ indicate training data from the chemical mapping, PseudoBase, and
PDB source collections. § indicates training or pretraining on bpRNA-1m,
RNAcentral, or Rfam because these collections overlap heavily.
