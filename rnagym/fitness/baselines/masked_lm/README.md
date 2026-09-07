# Masked-marginal scoring

For each mutated position, mask that base and score
`log p(mutant) - log p(wild type)`. Sum over the variant's mutations.
The four strategies differ in the context at the other mutated positions:

| strategy | fill at the other mutated sites | score |
|:--|:--|:--|
| `wt-fill` | wild-type bases | `sum_i log p(mt_i \| x^wt_-i) - log p(wt_i \| x^wt_-i)` |
| `mask-fill` | masks | `sum_i log p(mt_i \| x^wt_-M) - log p(wt_i \| x^wt_-M)` |
| `mut-fill` | mutant bases | `sum_i log p(mt_i \| x^mt_-i) - log p(wt_i \| x^mt_-i)` |
| `match-fill` | the allele being scored | `sum_i log p(mt_i \| x^mt_-i) - log p(wt_i \| x^wt_-i)` |

Here `M` is the set of mutated positions, `x^wt` the wild type, `x^mt` the
variant, and `x_-i` a mask at position `i`. All four agree on single mutants.

The leaderboard uses `wt-fill`, following the [ESM example][0] and
[ProteinGym baseline][1]. The [ESM paper][2] defines the other three contexts.
One run computes all four, sharing repeated contexts across strategies.

## Adding a model

Subclass `MaskedLMAdapter`, declare its alphabet, token limits and batch defaults,
and implement `add_arguments`, `load` and `logits_at`. Call `runner.main`.
[RNA-FM][3] is a short example. The engine checks the declared alphabet and
special tokens against the loaded tokenizer.

Long constructs are windowed around the masked span. A context fails if its
masked positions cannot fit within the model's sequence limit. Outputs include
one CSV per assay and a manifest with source, checkpoint arguments, batching and
hardware information.

Run `pixi run test` from `rnagym/fitness/` for the real-assay fixture checks,
or `pixi run -e rna-fm check-published` for checkpoint reproduction.

[0]: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/examples/variant-prediction/predict.py#L186-L225
[1]: https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/esm/compute_fitness.py#L486-L514
[2]: https://papers.nips.cc/paper_files/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html
[3]: ../RNA_FM/score_rna_fm_single_dms.py
