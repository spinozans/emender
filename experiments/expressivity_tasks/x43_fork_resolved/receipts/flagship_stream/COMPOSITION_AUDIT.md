# Stream composition audit — pile.txt source-order structure (no shuffle)

**Date:** 2026-11-09 · **Trigger:** operator question — "are we assuming pile.txt is
shuffled? because it ain't" · **Answer: no assumption of shuffle exists anywhere in
the program.** Both registrations say "source-order" explicitly
(PREREGISTRATION_1P3B.md line 98; byte_velocity_registration.json
`stream.policy: "source-order train/valid"`). The design's three invariants are
shuffle-independent: (1) span-disjointness (spine = [0,100e9), train/eval beyond) —
geometry; (2) within-board stream identity (all arms, same source-order documents) —
exactness of comparisons; (3) the document-boundary rule. This audit measures what
source-order actually means for composition.

## Method
Read-only `pread` samples from /mnt/nvme2n1/erikg/pile.txt. Coarse pass: 16KB at 8
offsets in each region. Dense pass: 8KB at 25MB spacing over 100.00–100.75GB.
Proxies: byte entropy, letter/digit/space fractions, markup/code character counts,
newline density, zlib-6 ratio.

## Findings

**Region A — byte-search/C14 training spans (0–59GB):** prose-dominant and uniform.
7 of 8 samples are plain prose (letters 0.75–0.81, entropy 4.5–4.8, markup ≈ 0);
one sample near 56GB shows code/structure. Effective composition: mostly
natural-language prose.

**Region B — flagship stream (100–101.5GB):** genuinely mixed, with structure.
Coarse pass found code-like (entropy 5.4, code-chars 0.098, zlib 0.12),
numeric/table (digits 0.06–0.08), and prose samples. Dense pass shows prose and
code/structured runs alternating at ~25–100MB granularity. More heterogeneous than
region A — but a *different* mix than A.

## Implications (receipt-driven, no claim amended)

1. **Within-board comparisons: unaffected.** All flagship/150M-lattice arms consume
   the identical source-order stream; arm-vs-arm and on/off-spine contrasts are exact
   regardless of composition.
2. **Copy-law: unaffected.** Span-disjointness is geometry, not statistics.
3. **Absolute numbers must be scoped:** flagship nats/byte are "the beyond-100GB
   source-order region's mix," not a Pile-average. State in the paper.
4. **Sharpened caveat — composition shift on hyperparameter transfer:** the byte
   searches tuned on prose-dominant region A; the flagship/150M arms train on mixed
   region B. The registered hyperparameter-transfer assumption now carries a
   measured composition shift. The 150M lattice (PREREGISTRATION_150M_LATTICE.md)
   is the empirical transfer test: arms 1–4 run A-searched configs directly on
   stream B — their dense-vs-dense numbers measure the shift at 150M before the
   1.3B commitment.
5. **Eval adjacency:** held-out eval documents are document-disjoint but adjacent in
   source order to training documents (both boards; the searches' valid windows are
   the first 24 adjacent valid documents). Without shuffle, held-out ≠ exchangeable:
   absolute scores read optimistically; arm comparisons unaffected (identical
   exposure).

## Reproducer
See `composition_audit_reproducer.py` (same directory). Read-only; offsets fixed;
no materialization.
