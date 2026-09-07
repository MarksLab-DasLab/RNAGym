These fixtures contain the complete Andreasson 2020 ribozyme, Domingo 2018 tRNA
and Tome 2014 GFP aptamer assays: 11,934 measured variants spanning all three
ncRNA leaderboard categories. The RNA-FM prediction files are copied byte for
byte from the predictions used by the current leaderboard and include all four
fill strategies.

The workflow test merges those released predictions and reproduces the metric
and fill-strategy tables. A separate checkpoint-free adapter exercises context
construction, batching, scoring, validation and manifest output on the complete
Domingo assay. No checkpoint or network access is required.

The EVmutation predictions use the complete released Andreasson MSA, fitted on
September 7, 2026. They test the comparison on covered variants and retain nulls
for variants outside the modeled positions.

`repeated/` contains the complete released Pitt 2010 ribozyme assay and its Evo2
7B predictions. It checks that merging retains all 186 measurements and uses
the first score for repeated mutations, matching the published leaderboard.
The expected Spearman correlation is frozen independently of the merge code.

`repeated/coding.csv` contains all 106 repeated measurements from the DOCK1,
HECD1, POLG_PESV, RCRO and SPTN1 Tsuboyama 2023 coding assays. Sequences and
measured fitness values are unchanged. GenSLM scores come from the full 2.5B
regeneration with native-loss checks on 48 rows per assay. These sequences have
distinct experimental fitness measurements that the merge must retain.
The fixture SHA-256 is
`1e2443617847f2fbe723c20200116f83648f14ac2edc1ce4d817c483407840ae`.

`fp8.csv` pairs published Evo2 1B/7B predictions with measured H100 inference
on all four reproduction inputs, totaling 986 variants per model. The source
CSVs were checked against their recorded SHA-256 hashes before combining them.
Both models use the production batch budgets. The comparison test accepts the
observed system variation and rejects larger errors and fitness-metric drift.
The fixture SHA-256 is
`d4812536691d1aa7d933656fa1e0b8c34f09d390b61b4b8d8e3ddae3b4a2761c`.

`evo.csv` contains the same 986 variants per Evo1/Evo1.5 checkpoint, scored on
H100 and L40S in both BF16 and float32. Checkpoint shards were verified against
their official Hugging Face hashes. With identical Python environments, BF16
changes prediction rankings and assay fitness correlations across GPUs.
Float32 passes the unchanged comparison bounds on all eight model/assay pairs.
The fixture checks both outcomes. Checkpoint revisions are `c206aab` for Evo1
and `99a9a4d` for Evo1.5, using Evo 0.5, PyTorch 2.7.1 and FlashAttention 2.7.4.
Float32 disables TF32 and uses the upstream unfused attention implementation.
The fixture SHA-256 is
`5b8c08fe822476e56628b4d5289a682be1f668fb8a2c5f2bf7f6becdb47ec050`.


`msa/Domingo_2018_tRNA.a3m` contains 1,000 unchanged records from the released
Riboseek alignment `sequence_0586129.a3m`. The selection is the first 250 records
plus the first 750 records at indices >= 1,000 with more than 90% canonical-base
coverage after removing A3M lowercase insertions. The companion FASTA is the
unchanged released query. Mixing these coverage levels exercises both covered
and excluded mutation sites in a bounded fit. The native regression expects
991 accepted sequences, modeled query positions 4 through 68, and 69 scored
variants among all 4,175 assay rows. Every remaining variant retains a null
score. Input and query SHA-256 hashes are:

- `Domingo_2018_tRNA.a3m`: `f78d13cd17dea318a14dc05e61621b2789ee5587dacfec1f6b58174479ac1cd6`
- `Domingo_2018_tRNA.fa`: `39795185d80ab9d8add329edd3b636a4fc2e5ca1011418a54440400b5b4527a4`
