# RNA3DB datasets for RNA Gym

These files were generated using the 2024-12-04 edition of [RNA3DB][1].

## Files

| Filename              | Description                                  |
| --------------------- | -------------------------------------------- |
| parse.json            | All RNA chains                               |
| filter.json           | Custom RNAGym filtering (>= 16 nt)           |
| cluster.seq.json      | Sequence-based clustering                    |
| cluster.struct.json   | Structure-based clustering                   |

## RNAGym Clustering

See [here][0] for full documentation and default settings.

1. Filter:
```bash
> python -m rna3db filter parse.json filter.json \
      --single_ratio_cutoff inf \
      --max_unknown_ratio inf \
      --max_resolution inf \
      --min_length 16 \
      --filter_log_path ./filter.log
```
2. Cluster (sequence only: 90% similarity, 90% identity)
```bash
> python -m rna3db cluster filter.json cluster.seq.json \
      --mmseqs_binary_path "$(which mmseqs)" \
      --min_seq_id 0.9 \
      --min_seq_coverage 0.9 \
      --mmseqs_coverage_mode 1 \
      --only_sequence
```
3. Cluster (structure only):
```bash
> # E-value cutoff of 1.00 to detect loose homology
> python -m rna3db cluster filter.json cluster.struct.json \
    --tbl_dir cmscans/ \
    --structural_e_value_cutoff 1.00 \
    --only_structure
```

## Citation

Marcell Szikszai, Marcin Magnus, Siddhant Sanghi, Sachin Kadyan, Nazim
Bouatta, Elena Rivas, RNA3DB: A structurally-dissimilar dataset split
for training and benchmarking deep learning models for RNA structure
prediction, Journal of Molecular Biology, Volume 436, Issue 17, 2024

<!--Links-->
[0]: https://github.com/marcellszi/rna3db/wiki/Documentation
[1]: https://github.com/marcellszi/rna3db/releases/tag/2024-12-04-full-release
