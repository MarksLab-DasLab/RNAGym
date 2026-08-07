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
Unsupported sequences receive zero. PseudoBase and PDB report each model's best
decoder. Macro is the unweighted mean of the three columns.

<!-- TODO(MCA): Add bpRNA-1m -->

## Leaderboard

<!-- BEGIN GENERATED TABLE -->
| Rank | Model | Chemical mapping (n=584k) | PseudoBase (n=358) | PDB (n=946) | Macro |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | RibonanzaNet*‡ | 0.4028 | 0.7683 | 0.8361 | 0.6690 |
| 2 | Vienna | 0.4480 | 0.7078 | 0.7555 | 0.6371 |
| 3 | EternaFold* | 0.4882 | 0.6175 | 0.7676 | 0.6245 |
| 4 | CONTRAfold | 0.4588 | 0.6091 | 0.7603 | 0.6094 |
| 5 | RNAstructure | 0.4429 | 0.6277 | 0.7553 | 0.6086 |
| 6 | UFold‡ | 0.3211 | 0.6526 | 0.7991 | 0.5910 |
| 7 | MXFold2 | 0.3809 | 0.5957 | 0.7907 | 0.5891 |
| 8 | RNA-FM‡ | 0.3053 | 0.6635 | 0.7948 | 0.5879 |
<!-- END GENERATED TABLE -->

\*, †, and ‡ indicate training data from the same source collection as the
chemical mapping, PseudoBase, and PDB benchmarks, respectively.
