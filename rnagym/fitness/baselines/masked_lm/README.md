# Masked-marginal scoring

Shared scoring code for the benchmark's masked RNA language models. A model's
script in `rnagym/fitness/baselines/<MODEL>/` supplies an alphabet, a tokenization and
one forward pass. Everything else lives here.

## The four fill strategies

Every masked-marginal score computes a log-odds at each mutated position of a
variant and sums over its mutations. The strategies differ in exactly one thing:
what fills the variant's OTHER mutated positions while the scored position is
masked. With `M` the mutated positions, `x^wt` and `x^mt` the wild-type and
variant sequences, `x_-i` a mask at `i` and `x_-M` masks at every position in `M`
([Meier et al. 2021][0], supplement Appendix A):

| strategy | fill at the other mutated sites | score |
|:--|:--|:--|
| `wt-fill` | wild-type bases | `sum_i log p(mt_i \| x^wt_-i) - log p(wt_i \| x^wt_-i)` |
| `mask-fill` | masks | `sum_i log p(mt_i \| x^wt_-M) - log p(wt_i \| x^wt_-M)` |
| `mut-fill` | mutant bases | `sum_i log p(mt_i \| x^mt_-i) - log p(wt_i \| x^mt_-i)` |
| `match-fill` | the allele being scored | `sum_i log p(mt_i \| x^mt_-i) - log p(wt_i \| x^wt_-i)` |

Appendix A defines `mask-fill`, `match-fill` and `mut-fill`. `wt-fill` is the
closely related convention implemented under the name `masked-marginals` by the
[ESM example][1] and [ProteinGym baseline][2].
The name "masked marginals" is overloaded. `wt-marginals` is a different method
again: one unmasked forward pass with no masking.

All four are identical on single mutants, because a variant with one mutation has
no other mutated positions. They diverge on multi-mutants, which are 99.4% of the
ncRNA benchmark. `match-fill` alone mixes two contexts, so it is a
difference of two conditionals rather than a log-odds ratio.

## Cost

One run computes all four. Their contexts overlap, so on the 31 ncRNA assays
the four together need 2,929,196 unique context examples against 2,458,521 for
`mut-fill` alone, about 19% more. Computing each strategy separately would cost
more because their shared contexts would be rerun. `match-fill` must be
accumulated directly because it cannot be reconstructed from the final
`wt-fill` and `mut-fill` scores.

## Files

| file | contents |
|:--|:--|
| `strategies.py` | mutation parsing, wild-type recovery, per-variant validation, the four strategies as contexts and terms |
| `engine.py` | windowing, batching, the gather, and accumulation |
| `adapter.py` | the model interface |
| `runner.py` | command line, assay input, wild-type cross-check, output and manifest |

## Sequence length

RNA-FM and RiNALMo cap the number of positions they accept. The ncRNA assays
are 45 to 425 nucleotides and never reach it, but the mRNA constructs are
kilobases. Each long context is windowed around its masked span. A context cannot
be scored when its masked positions span more than the model's sequence limit.

## Adding a model

Subclass `MaskedLMAdapter`, declare the alphabet, output column, special-token
count and batching defaults, implement `add_arguments`, `load` and `logits_at`,
and call `runner.main`. `rnagym/fitness/baselines/RNA_FM/score_rna_fm_single_dms.py` is
the shortest example. Two rules the engine enforces rather than assumes: the
declared special-token count must match the loaded tokenizer, and context
positions must map to token positions by a constant shift.

## Output

One CSV per assay, holding the assay dataframe plus `{column}_{strategy}` for
each strategy computed, and a `{DMS_ID}.manifest.json` recording the strategies,
alphabet, model arguments, hardware, counts and a hash of the scoring source.
Requesting a single strategy also writes the model's historical bare column, so
existing commands keep producing the files they used to.

## Tests

`tests/test_fitness.py` runs the scorer, manifest writer, prediction merge,
fill-strategy analysis and benchmark aggregation on three complete released
ribozyme, tRNA and aptamer assays. A deterministic stand-in model keeps the test
CPU-only and checkpoint-free. Checkpoint-backed `predict` and
`check-published` tasks run in the model environments defined in
`rnagym/fitness/pixi.toml`.

[0]: https://papers.nips.cc/paper_files/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html
[1]: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L186-L225
[2]: https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/esm/compute_fitness.py#L486-L514
