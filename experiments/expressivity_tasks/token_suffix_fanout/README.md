# Lookup that drives tokens

A reference for the operator's token-space version of `BRANCH`: **a read returns the weighted choices for the next token and a handle for each exact continuation.** It doesn't stop at a vector of next-byte logits. A token-level predictor could use these counts directly as a prior, or attention could reweight the alternatives and save a handle for a later read. This directory implements the index and a spelling adapter, **not** a fitted predictor or a native P100 token index.

## A distinction visible on six invented documents

```sh
.venv/bin/python -m experiments.expressivity_tasks.token_suffix_fanout.demo
.venv/bin/python -m pytest -q experiments/expressivity_tasks/token_suffix_fanout
```

The demo tokenizes raw bytes with a tiny, reversible, prefix-free vocabulary. A token commits on its last byte; no later byte decides where it ended. Each archive document resets the text boundary. Counts here are invented, not measurements of English.

| Observed token suffix | Outgoing token counts | Next byte |
|---|---|---|
| `New` | `" York"`: 2, `" Jersey"`: 1 | space: 3 |
| `Old` | `" York"`: 1, `" Jersey"`: 2 | space: 3 |

The first-byte forecast is **identical**. Once a real space arrives, the *saved fanouts*, with no new lookup, predict `Y:J` in proportions **2:1** vs **1:2**. Complete `" York"` after `New` and the selected child represents `New York`: `" City"` has count 2. Relax to its suffix `" York"` and `" Road"` also appears (count 1). Following `" Road"` from `New York` would be an absent **exact** branch; observing it for real instead backs off to the matching `York Road` suffix. Exploring a branch does not change the observations. None of this requires an occurrence list or a speculative read of each child's contents.

## What the reference guarantees

- [`index.py`](index.py) builds a static suffix automaton **over integer tokens**, not their serialized bytes. `seek(observed)` finds the longest matching token suffix. `branch(locus)` lists all distinct followers, their exact multiplicities and child handles; document ends are separate. The edge count is the number of corpus end positions for the extended string. One occurrence contributes one count. When keys and values are tied per token, attention can weight each token once with `log(count) + score(token)` instead of repeating its row for every occurrence. It cannot do this for arbitrary occurrence-specific values.
- `follow` is an exact hypothetical right extension (no silent fallback). `advance` consumes a real token, using suffix links to back off on a miss. `suffix` drops **one leading token**, not the last token and not necessarily a whole automaton state. A handle includes its mount and its matched token length: different suffix lengths can share a state and the same fanout.
- `spelling_step(answer, pieces, observed_since_read)` uses only the returned page and the bytes actually observed since. It reports next-byte counts, completed token edges and structural end mass **separately**. If `a` and `ab` are both tokens, after `a` the first branch is ready for another token while the second still predicts `b`. Don't discard the completed branch, certify a boundary from the unseen target, or renormalize the unfinished mass and call it the full next-byte law. If the supplied bytes already pass a candidate boundary, `needs_successor` returns its edge and the as-yet-unverified trailing bytes, rather than silently dropping that path. Equal spellings need not mean equal token IDs. The reference never consults another node on its own.
- [`test_index.py`](test_index.py) compares the index to independent document-local occurrence enumeration, including overlaps, repeats, unknown tokens and lookups longer than eight tokens. It checks direct vs backed-off extension, exact suffix lengths, structural ends, foreign handles, the same-first-byte fork and spelling collisions. The Python dictionaries and in-memory build are an oracle for small inputs, **not** a proposed 100 GB serving format.

## What to test on real data, separately

[**X53 now tested the full-answer vs first-byte comparison on real text.**](../x53_token_spelling/RESULT.md) General spelling knowledge helped substantially, and the suffix node’s specific continuation frequencies added a smaller, measurable gain. That separate study used a new BUILD-only token index, not P100.

1. Fix the tokenizer, decoder, source identity and document boundaries. The mounted P100 directory indexes raw bytes. Looking up a token's spelling there is **not** an exact token-aligned follower count. A token index over source documents, or a proven boundary- and token-ID-aware view of the same source, is needed. A node ID alone is not a portable token or a permanent address across builds. If the index itself supplies variable-length phrases as tokens, specify an online parsing/commit rule: that's a distinct choice from a fixed vocabulary.
2. To isolate **suffix information**, use the same tokenizer and token exposure for a root-frequency reader and a contextual token-fanout reader. To isolate **using an answer over several bytes**, hold the actual lookup sites and source fixed; compare full token alternatives against the *same answer collapsed to its first-byte histogram*. A replay of the original byte forecast cannot distinguish the two rows above. To price **handles**, compare following the selected edge with seeking the same extended string from scratch; they should give the same answer, not magically different information.
3. Keep raw external bytes as observations/targets for the byte-reader comparison. Require causal boundary decisions or carry incomplete/ambiguous paths without revealing future segmentation. Include a registered fallback for unsupported bytes/tokens; score all bytes, reset at document boundaries. Token NLL divided by decoded length for one canonical encoding is not by itself a demonstration of an online conditional next-byte predictor. Count tokenizations, real archive calls, fanout entries and transported bytes, and measure end-to-end time per original byte. Fewer calls per byte need not be cheaper if the alphabet or page is much wider.

No tokenizer fit, natural data reservation, archive rebuild, training or scale result is part of this reference. [X52](../x52_sparse_read/RESULT.md) tested regular sparse **byte** lookups, not token fanout.
