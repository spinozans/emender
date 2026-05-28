# Editorial Notes Paper Synthesis

Date: 2026-05-28 UTC
Task: `editorial-notes-paper-synthesis`

This note records the final synthesis pass after
`editorial-notes-focused-edit`. It is a validation artifact, not a new
editorial rewrite.

## Scope Checked

- Compared the stocktake and focused-edit surfaces against `paper/main.typ` and
  `paper/refs.bib`.
- Scanned the live paper/release surfaces for naming, release links, Figure 2
  values, proof-boundary language, and current concurrent-work framing.
- Kept the focused edit boundaries intact: five core suggestions, the Section 5
  racer/Pareto reframe, the GDN-2/Mamba-3 related-work note, bounded
  update-rule freedom, S5 length-degradation honesty, and mechanical
  delta-notation fixes.

## Synthesis Fixes

- Updated the root `README.md` memory-update sketch so the public description
  matches the paper's production E88 recurrence: SiLU-activated values in the
  delta term, L2-normalised SiLU keys/queries, and the SiLU output gate.
- Updated `paper/README.md` to stop describing Figure 2 as pending 1.27B data;
  it now points to the staged 1.3B-class Figure 2 plot and the current BPB
  snapshot.

## Confirmed Invariants

- Figure 2 values remain E88 `0.979`, GDN `0.975`, and M2RNN-CMA `0.984`.
- The live paper and release hub use the `poietic-pbc` namespace and the
  `1.3b` Hugging Face slugs:
  `poietic-pbc/emender-e88-1.3b`, `poietic-pbc/gdn-1.3b`, and
  `poietic-pbc/m2rnn-cma-1.3b`.
- The paper preserves exact parameter counts: E88 `1.273 B`, M2RNN-CMA
  `1.307 B`, and GDN `1.352 B`.
- GDN-2 and Mamba-3 are cited as concurrent linear-state work and kept outside
  the matched racer; the paper does not claim that either was evaluated here.
- Widened TC0/transformer language was not added. The paper keeps explicit
  non-claims around exceeding NC1/TC0 and states Lean proof boundaries before
  the formal theorem list.
- Stale `1.27b`, `poieticpbc`, and old release-target strings were not found in
  the live release surfaces checked. Remaining `1.27B`/`1p27B` occurrences are
  historical notes or exact Lean theorem identifiers.

## Validation To Record In `wg log`

- `bash paper/build.sh` exits 0 with no Typst warnings in stdout/stderr.
- `cd formal/lean && ./scripts/check_paper_core.sh ElmanProofs/PaperCore.lean`
  passes.
- The rebuilt generated PDF was spot-checked around the abstract, Section 5
  Figure 2, related work, conclusion/release paragraph, and references.
- The generated PDF was uploaded to `erik@hypervolu.me:~/www/ndm/` and verified
  at the public URL.
