# RNAGym 2D Leaderboard

| Rank | Model | Chemical mapping (n=585k) | PseudoBase (n=358) | PDB (n=946) | Macro |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 1 | RibonanzaNet*‡ | 0.3330 | 0.7683 | 0.8361 | 0.6458 |
| 2 | Vienna | 0.3811 | 0.7078 | 0.7555 | 0.6148 |
| 3 | EternaFold* | 0.4211 | 0.6175 | 0.7676 | 0.6021 |
| 4 | CONTRAfold | 0.4029 | 0.6091 | 0.7603 | 0.5908 |
| 5 | RNAstructure | 0.3842 | 0.6277 | 0.7553 | 0.5890 |
| 6 | RNA-FM‡ | 0.3080 | 0.6635 | 0.7948 | 0.5888 |
| 7 | UFold‡ | 0.2971 | 0.6526 | 0.7991 | 0.5829 |
| 8 | MXFold2 | 0.3064 | 0.5957 | 0.7907 | 0.5643 |

Chemical mapping is the macro mean Spearman across modifiers. PseudoBase and
PDB are the best cluster-macro F1 across each model's decoders. Macro is the
unweighted mean of the three columns. Underlying values are in
[`leaderboard.csv`](leaderboard.csv).

\*, †, and ‡ indicate training data from the same source collection as the
chemical mapping, PseudoBase, and PDB benchmarks, respectively.
