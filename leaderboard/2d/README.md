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
| 1 | RibonanzaNet*†‡§ | 0.3898 | 0.7703 | 0.8479 | 0.5527 | 0.6881 | 0.6497 |
| 2 | Vienna | 0.4270 | 0.7131 | 0.7791 | 0.5181 | 0.7276 | 0.6330 |
| 3 | EternaFold* | 0.4685 | 0.6173 | 0.7849 | 0.5392 | 0.7442 | 0.6308 |
| 4 | RNAstructure | 0.4218 | 0.6295 | 0.7785 | 0.5129 | 0.7345 | 0.6155 |
| 5 | CONTRAfold | 0.4373 | 0.6128 | 0.7783 | 0.5361 | 0.7120 | 0.6153 |
| 6 | UFold†‡§ | 0.3008 | 0.6572 | 0.8087 | 0.6666 | 0.5713 | 0.6010 |
| 7 | MXFold2†‡§ | 0.3614 | 0.5976 | 0.8037 | 0.5257 | 0.6667 | 0.5910 |
| 8 | RNA-FM†‡§ | 0.2867 | 0.6695 | 0.8037 | 0.5923 | 0.5052 | 0.5715 |
| 9 | RiNALMo†‡§ | 0.2584 | 0.4615 | 0.6878 | 0.7682 | 0.2195 | 0.4791 |
<!-- END GENERATED TABLE -->

\*, †, and ‡ indicate training data from the chemical mapping, PseudoBase, and
PDB source collections. § indicates training or pretraining on bpRNA-1m,
RNAcentral, or Rfam because these collections overlap heavily.
