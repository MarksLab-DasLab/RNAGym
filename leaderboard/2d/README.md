# RNAGym 2D Leaderboard

| Rank | Model | Chemical mapping | PseudoBase | PDB | Macro |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | RibonanzaNet*‡ | 0.3330 | 0.7613 | 0.8358 | 0.6434 |
| 2 | EternaFold* | 0.4211 | 0.5988 | 0.7632 | 0.5944 |
| 3 | RNA-FM‡ | 0.3080 | 0.6573 | 0.7740 | 0.5798 |
| 4 | UFold‡ | 0.2971 | 0.6456 | 0.7883 | 0.5770 |
| 5 | Vienna | 0.3811 | 0.5859 | 0.7525 | 0.5732 |
| 6 | CONTRAfold | 0.4029 | 0.5837 | 0.7315 | 0.5727 |
| 7 | RNAstructure | 0.3842 | 0.5857 | 0.7434 | 0.5711 |
| 8 | MXFold2 | 0.3064 | 0.5957 | 0.7907 | 0.5643 |

Chemical mapping is the macro mean Spearman across modifiers. PseudoBase and
PDB are cluster-macro F1 scores. Macro is the unweighted mean of the three
columns. Underlying values are in [`leaderboard.csv`](leaderboard.csv).

\*, †, and ‡ indicate training data from the same source collection as the
chemical mapping, PseudoBase, and PDB benchmarks, respectively.
