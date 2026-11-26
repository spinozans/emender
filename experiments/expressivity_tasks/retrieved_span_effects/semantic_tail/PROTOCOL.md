# No-hit corpus read: can the trained token head follow a two-fact edit?

**Invented-token, single-thread CPU experiment.** No GPU, natural text, existing NLP weights or previous cohorts. Commit this protocol and the tested implementation before any fits. This is a mechanism check on the existing [corpus head](../corpus_kernel/README.md) and [packed training path](../corpus_windows/README.md), not a new language-quality panel or a requalification of the failed GPU meter.

## Question and comparisons

Every query token is outside the archive vocabulary in use, so even a length-one MEM is impossible. Can the **nonzero background over all occurrences** carry useful learned information, not just nonzero normalizing mass? Swap two answers, change no prompt, and ask if the frozen model follows the new bindings. Compare three separately co-trained vector heads:

- **LOCAL:** only the causal live rows and base/no-read row.
- **DENSE:** all archive rows at exact dot-product scores, plus the same kinds of local and no-read rows.
- **INDEXED:** the same rows and priors, no exact refinements, using the existing positive Gaussian feature summaries. Rank 32, antithetic bank seed 5700. No early return on an empty posting list.

Use the existing `CorpusTokenLM`/`Windows.backward_nll`, not a fixture-specific copy classifier. All three have the same trainable modules, start weights within seed, loss, target order and dose. The corpus value is a learned vector and the LM must decode it; it is **not** a literal successor-token pointer. Row log priors zero, no-read score includes the head's existing `log(max(1, archive_rows))` offset. DENSE vs INDEXED retain identical source priors. LOCAL is a fitted corpus-absent baseline, not a fixed-mass value ablation.

## Fixed task and split

Eight two-token documents per edition, `[key_i, value_π(i)]`, hence exactly eight predictive boundaries, one per key. Values in each edition are a permutation: all eight occur exactly once. The associated live document is `[prompt, alias_i, value_π(i)]`; predict **both** shifted targets. IDs: prompt 0; key `1+i`; alias `9+(3*i+5)%8`; value `17+j`. Disjoint token roles, no edition ID as model input, no memory between live documents. At the answer frontier there is one causal local row, whose successor is the already observed *alias*, never the answer. Only the external value binding distinguishes edits.

Group permutations into orbits under adding an offset modulo eight to every value; canonical form has `π(0)=0`. Order the 7! canonical forms by SHA256 of ASCII `semantic-tail-v1:` followed by the eight value bytes. Use the first 16 orbits × all 8 offsets = **128 FIT editions**. For each of the 28 unordered key pairs in lexicographic order, take the first unused canonical form such that it and its two-key transposition belong to two different unused orbits; reserve both orbits. Probe all eight offsets, giving **224 original/edited edition pairs**, 448 edited bindings, 1,344 unchanged bindings. No probe orbit appears in FIT; individual facts, aliases and templates recur deliberately. Within each orbit each alias sees each value equally often. The held-out unit is a complete binding permutation, not an unseen word or a new language. All choices deterministic without outputs or selection by loss.

## Training and freeze

Three paired seeds 5701–5703. Width 32, learned vector read, leaf size four, CPU FP32, one thread. For each of eight epochs permute the 128 FIT editions with a continuous NumPy `default_rng(57100+seed)`. An update takes all eight reset live windows of one edition in key order: 16 real shifted targets, blocks of five, global mean NLL. Encode under current weights on **every** update, including archive rows; cache only IDs and routes. AdamW `.003`, betas `.9/.999`, decay `.01`, clip norm 1. Constant LR, 1,024 updates / 16,384 real shifted targets per fit. No schedule selection, retries, rank sweep or warm start. Record actual steps, exposures, price, initial/order/final hashes and gradient coverage. Fit and freeze **all nine** final snapshots before probe scoring. Verify the source, freeze manifest, identities, bytes and order receipts before evaluating any probe case. No checkpoint selection by probe. Failures and seed reversals stand.

## Primary and diagnostics

Report each trained reader separately. A reader **passes this invented binding test** only if in **each** seed: (1) its native edited-edition answer NLL over all eight keys beats separately trained LOCAL; (2) more than half of the 448 changed-key pairs have the *old* answer top-1 before the edit **and** the *new* answer top-1 after it, on identical live inputs; (3) its new-answer NLL on the changed keys is lower with the edited archive than with the original (stale) archive at the **same** weights. Native = training law. Larger changes in logits or corpus share alone do not pass. Do not transfer a DENSE pass to INDEXED or vice versa.

Also report old/edited answer accuracy, changed and unchanged strata, stale vs intact, cue and all-target NLL, corpus/local/no-read shares at answer positions, and probability/argmax of the intended key under the conditional row law. Re-score both trained corpus heads on edited editions under the other law at fixed weights; report new NLL, row-law and next-token total variation. This separates learning an effective approximate read from faithfully approximating the dense read. Enumerate the entire prespecified probe; seed values and exact counts, no claim of independent natural-document confidence intervals. An LM can potentially decode a mixture without putting the most weight on the intended row: do not equate output accuracy with row accuracy. No scale or throughput extrapolation from eight rows.
