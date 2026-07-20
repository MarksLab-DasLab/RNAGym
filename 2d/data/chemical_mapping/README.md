# RNAGym 2D datasets

| Dataset | Contents |
| --- | --- |
| `rnagym_2d.parquet` | 947k chemical mapping profiles for 585k sequences, split into train and test. |
| `rnagym_pseudobase.parquet` | 360 sequence/structure pairs with pseudoknots |


## Reproducing the datasets

`process_raw.py` builds two datasets from `raw_data/`. From this directory, to
reproduce:

```bash
# If already in `rnagym2d` env
./process_raw.py

# Alternatively,
pixi run ./process_raw.py
```

Chemical mapping sequences are clustered with MMseqs2 at 40% sequence identity
and 80% coverage. Clusters are split 80/20 into train/test with random seed 42.

See [`raw_data/README.md`](raw_data/README.md) for sources and schemas.
