# X53 — full token branches versus their first byte

[Protocol](PROTOCOL.md) · [token suffix reference](../token_suffix_fanout/README.md) · [previous sparse-byte result](../x52_sparse_read/RESULT.md)

The comparison isolates one piece of a lookup: knowing how the candidate tokens **continue** after their first byte. FIRST, LEX and SPELL share the actual query sites, suffix nodes, byte predictor, root backoff and full first-byte probability vector. FIRST gets no lexical evidence until the next lookup; LEX knows general token spellings; SPELL retains the original node's spelling preferences. All programs score ordinary shifted bytes, not token IDs. This uses a new small BUILD-only index, not the byte-indexed P100 container.

- `archive.py`: reversible complete-token spelling rule, BUILD byte/lexical counts, private numeric image, suffix-automaton mount and packed response service.
- `reader.py`: complete causal byte distributions and spelling-prefix range sums. Updating the local spelling prefix does not ask the suffix index for another page.
- `study.py`: independent program paths with rotated block order, audit of common predictions/navigation, whole-document paired intervals with band-specific byte denominators and accounting for actual copied response sizes.
- `data.py`, `run.py`: publish qualification before selection; verify the spent rank chain; witness a frozen count image and compiled index before PROBE opens. No gradient optimization or automatic retry.
- `test_fanout.py`: independent occurrence enumeration and explicit byte-count predictions at every prefix, structural ends, no-gap-query checks, support and information-boundary tests.
