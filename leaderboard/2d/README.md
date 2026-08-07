# RNAGym 2D Leaderboard

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

Chemical mapping is the macro mean Spearman across 1M7, 2A3, DMS, and NMIA.
BzCN has low replicate agreement, while CMCT has only 11 replicate-bearing
clusters. These and the degradation assays are reported only in
[`leaderboard.csv`](leaderboard.csv). PseudoBase and PDB are the best
cluster-macro F1 across each model's decoders. Macro is the unweighted mean of
the three columns.

\*, †, and ‡ indicate training data from the same source collection as the
chemical mapping, PseudoBase, and PDB benchmarks, respectively.
