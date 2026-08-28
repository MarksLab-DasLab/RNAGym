These fixtures contain the complete Andreasson 2020 ribozyme, Domingo 2018 tRNA
and Tome 2014 GFP aptamer assays: 11,934 measured variants spanning all three
ncRNA leaderboard categories. The RNA-FM prediction files are copied byte for
byte from the v0.2 prediction release and include all four fill strategies.

The workflow test merges those released predictions and reproduces the metric
and fill-strategy tables. A separate checkpoint-free adapter exercises context
construction, batching, scoring, validation and manifest output on the complete
Domingo assay. No checkpoint or network access is required.
