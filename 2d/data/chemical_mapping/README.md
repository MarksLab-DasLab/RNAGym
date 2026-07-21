# RNAGym 2D datasets

| Dataset | Contents | Unique key |
| --- | --- | --- |
| `rnagym_2d.parquet` | 947k chemical mapping profiles for 585k sequences, split into train and test. | `seqID` |
| `rnagym_pseudobase.parquet` | 360 sequence/structure pairs with pseudoknots | `pseudobase_ids` |


## Reproducing the datasets

To reconstruct both datasets from `raw_data/`:

```bash
pixi run collate-2d
```

Chemical mapping sequences are clustered with MMseqs2 at 40% sequence identity
and 80% coverage. Clusters are split 80/20 into train/test with random seed 42.

See [`raw_data/README.md`](raw_data/README.md) for sources and schemas.
