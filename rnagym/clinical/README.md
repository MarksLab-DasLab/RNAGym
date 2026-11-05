# RNAGym clinical variant benchmark

Zero-shot scoring of pathogenic and benign human variants with RNA and DNA
language models. ClinVar variants (two or more review stars) are read on four
views: SNVs and indels, each on the pre-mRNA and on the spliced mRNA.
Mitochondrial variants from MITOMAP are read on the processed RNA. Results are
on the [leaderboard][0]; the files are described in [data/clinical][1].

## Scores

| Models | Score |
| --- | --- |
| Masked: RNA-FM, RiNALMo, RNAGenesis, AIDO.RNA (5 sizes), Nucleotide Transformer v3 (3 sizes), Orthrus | SNVs: the masked marginal at the variant, `log p(alt) - log p(ref)` with the site masked in the reference window. Other variants: the mean pseudo-log-likelihood of every position within 50 nt of the variant, alternate minus reference |
| Autoregressive: Evo 2 (1B base, 7B, 20B, 40B) | Mean log-likelihood per token of the alternate window minus that of the reference window, forward strand, every window scored alone |

Each model reads a window centred on the variant, at its own context limit and
at most 10,101 nt: 1,022 nt for RNA-FM, RiNALMo and AIDO.RNA, 2,048 for
RNAGenesis, 8,192 for Evo 2 1B and 10,101 for the others (`models.py`). A
variant whose alleles do not fit the window is not scored. Higher scores mean a
more likely alternate allele, so the evaluation negates them.

## Evaluation

AUROC is computed within each consequence class with at least 10 pathogenic and
10 benign variants, on the variants every model scored. A unit's macro averages
its classes, and the ClinVar score averages the four units. The mito score
averages the tRNA AUROC, on the tRNA with its CCA tail, and the mRNA missense
AUROC. Intervals come from 1,000 bootstrap draws of genes (mitochondrial RNA
units), the same draws for every model.

## Run

Data come from the RNAGym data archive (see the repository README). Steps run
from this directory:

```bash
pixi run prepare                     # stage data/clinical/variants
NUM_SHARDS=24 pixi run score evo2_7b premrna_10101_snv   # one shard per array task
pixi run merge                       # every scored file and model
pixi run evaluate                    # the six files, 1,000 draws
pixi run leaderboard                 # leaderboard/clinical
```

`score` runs each model in its environment of the fitness benchmark
(`../fitness/pixi.toml`), after the same setup as the fitness scoring scripts:
weights for RNA-FM, RiNALMo and RNAGenesis, the RiNALMo source, and each Hub
model's cache under `RNAGYM_CHECKPOINT_DIR`. The shard is `SLURM_ARRAY_TASK_ID`
of `NUM_SHARDS`; rows sharing a reference sequence stay in one shard. Every
shard is written with a record of its scoring spec, package versions,
checkpoint hashes and source hash, and `merge` refuses shards that are missing,
mixed or scored from another staged file. Local checkpoint hashes are cached in
`.cache/clinical/checkpoint_sha256.json`; delete it to hash them again.

To rebuild the leaderboard from the released scores, skip `score` and `merge`;
evaluating the pre-mRNA SNVs takes about 20 minutes.

## Quality checks

```bash
pixi run test
pixi run lint
pixi run fmt
```

The tests check the scorers against fake models computed by hand, staging,
sharding and merging, and run the evaluation and the leaderboard on a fixture of
real variants and scores (`tests/fixtures/clinical`).

[0]: ../../leaderboard/clinical/README.md
[1]: ../../data/clinical/README.md
