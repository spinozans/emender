# REWRITE_AUDIT.md — the gate (READ-ONLY whole-document verification)

**Subject:** Garrison, *Emending Nonlinear Recurrence* (`paper/main.typ`, 2667 lines)
**Mode:** READ-ONLY. `main.typ` was NOT edited. This is the post-rewrite gate before
the result reaches the author.
**Inputs:** full `paper/main.typ`; `paper/review/REWRITE_PLAN.md` (spine + 4 mandatory
reconciliations); `paper/review/EXECUTIVE_SUMMARY.md` (settled scope, Track-A/B ledger).

---

## VERDICT: **RESIDUAL (minor / near-ship)**

The rewrite is sound on every gating axis and is shippable after a handful of
cosmetic, pointer-level touch-ups. Specifically:

- **NUMBER / CLAIM INTEGRITY — PASS (no drift).** Every load-bearing value in the task
  list survives unchanged and in its correct scope across all 12 passes. Zero altered
  values, zero dropped claims, zero numbers stated out of scope.
- **FOUR MANDATORY RECONCILIATIONS — ALL LANDED** (one with a soft note: a condensed
  BL-1 restatement remains in §9 as a deferring pointer rather than being fully deleted).
- **VOICE — PASS.** The abstract+§1 reader-first / outside-in register holds all the way
  down. No section drifts back to inside-the-project voice.
- **R1/R2/R3/R4 RESIDUAL STANCES — effectively purged.** The R2 "production / deployed"
  ontology is **gone** (0 hits). R1 codenames are defined-at-first-use or glossed. R3
  pre-litigation is confined to the scope structure. No R4 process/QA leaks.

It is **not REGRESSED** (no number drifted, no reconciliation failed, voice did not break)
and **not strictly CLEAN** only because a small number of minor seams survive, listed
below per the task's "list spans + rule + section" requirement.

---

## 1. NUMBER / CLAIM INTEGRITY SWEEP (highest priority) — PASS

Every load-bearing value confirmed, with line citations and scope check:

| Value | Confirmed at | In correct scope? |
|---|---|---|
| **bpb 0.973** (E88 train bpb) | 310, 401, 1090, 1111, 1124, 2257 | ✓ train-loss bpb, E88 viability; never used as cross-arch verdict |
| **0.966 / 0.966 / 0.961** (held-out bpb, E88/GDN/M²RNN-CMA) | 407, 1151–1152, 1184, 2617 (0.96613/0.96612/0.96132), 2634–2636 | ✓ held-out, framed as statistical tie |
| **Held-out tie** (0.006 band < 0.025 cross-slice σ; lowest-counts 0/5,1/5,4/5) | 1163–1175, 1184 | ✓ matches EXEC_SUMMARY (0.0058 band < 0.0254) |
| **22,200 programs/token** (370×5×12) | 313, 624, 687, 716, 752, 785, 904, 2262 | ✓ consistent everywhere |
| **MFU ~15.7%** | 928, 933 (~16%) | ✓ occupancy-vs-MFU, conservative-lower-bound framing |
| **~91% of FLA-GDN** (7,492 / 8,248 tok/s) | 940–947 | ✓ parity-class, not superiority |
| **S5 8M acc 0.79 / 0.36 / 0.22** | 259–261, 342–343, 826, 833–834, 1284–1286, 1321–1322 | ✓ Emender/GDN/M²RNN-CMA at T=128 |
| **1.3B ordering** (E88 > M²RNN > GDN at every length) | 1448–1450, 1465, 1507, 1529, 1560–1561 | ✓ ordering preserved, scoped to efficiency |
| **0.143 plateau** (E88 S5 T=512, 1.3B) | 1448, 1466, 1530, 1563 | ✓ "plateau below ceiling", flat from ~12k steps |
| **1.273 B params** | 246, 353, 621, 1025, 2634 | ✓ |
| **S5 length series 0.79 / 0.42 / 0.22 / 0.11** (8M Emender, T=128/256/512/1024) | 1325–1326, 2155–2156 | ✓ consistent §6 ↔ §9 |

Supporting values also cross-checked consistent: train-loss 0.973/0.973/0.979 (1125, 1153,
1184); 8M S3 controls 1.00/0.72/0.31 (1284–1286, 1321–1322); 1.3B S5 symmetric series
0.921/0.536/0.272/0.143/0.076 (E88) and 0.658/0.335/0.172/0.090/0.049 (M²RNN) and
0.117/.../0.022/0.015 (GDN) (1448–1450); to-competence E88 0.162 @ T=512 (1428, 1509);
appendix readback ≤2×10⁻⁵ nats (2617).

**No number drifted; nothing dropped; nothing stated out of scope.**

---

## 2. THE FOUR MANDATORY RECONCILIATIONS — verification

**(a) Held-out bpb = statistical TIE/parity; train-loss ordering retired as a verdict.
LANDED.** §5 subsection "Held-out bits-per-byte: the three are a statistical tie"
(1142–1175): "Critically, the train-loss ordering does *not* reproduce held-out … the three
update rules are a *statistical tie*". The loss-vs-wallclock figure caption explicitly
demotes the train curve ("a diagnostic of training dynamics, not the basis for the
architecture comparison — the reported measurement is the held-out bpb", 1085–1088).
Propagated backward to the abstract ("statistically tied, so language-modeling loss does
not distinguish them", 86–88), §1 (211–213), claims-table row 2 ("Does not license the
train-loss ordering as an architecture verdict", 411), and conclusion (2295–2300). ✓

**(b) §6 "matched no-tuning" relabeled + BL-1 disclosure asymmetry foregrounded in
Expressivity; duplicate removed from Limitations. LANDED (with a soft note).**
- Relabel: §6 heading is now "Matched no probe-tuning, and the one selection asymmetry it
  does not cover" (1226). ✓
- Foregrounding: the full BL-1 statement lives in §6 (1238–1265) — tanh tied on LM loss
  and kept on a state-tracking proxy, the S₃ control + §7 NC¹ theorem bounding
  direction/mechanism but not magnitude. ✓
- **Soft note:** §9 Limitations still carries "Design-space asymmetry of the 8 M defaults"
  (2190–2204), a ~14-line condensed restatement. It now explicitly defers ("§6 states this
  selection asymmetry in full", 2197), so it functions as a limitations-pointer, not a
  competing primary location — content-consistent, no contradiction. But against the literal
  letter of reconciliation (b) ("duplicate removed from Limitations") it is a *condensation*,
  not a *removal*. Defensible editorially; flag for author to decide.

**(c) Throughput = occupancy + MFU ~15.7%, NOT peak-FLOPs. LANDED.** §4 "Measured
throughput and utilization" (911–947): median 100% util / 97% power cap stated as
**occupancy**, explicitly separated from arithmetic ("This figure is occupancy, not peak
arithmetic, and the two should not be run together", 925); MFU 15.7% as a conservative
lower bound; width-axis parity at ~91% of a real FLA-GDN linear-scan kernel; closes "the
standing is parity-class, not superiority" (947). No "saturates the GPU" / "full
utilization" overclaim survives (0 hits). ✓

**(d) S₅ = efficiency-not-impossibility; never "solves S₅". LANDED.** Claims-table row 3
non-claim: "Not 'delta solves / length-generalizes S₅'… an efficiency gap, not
impossibility" (421–423). §6 "Trained-length competence vs length-extrapolation plateau"
(1521–1537) and "Delta vs raw-write: a budget-robust ordering, not solve-vs-fail"
(1551–1568) and "The delta update is also length-bounded" (1570–1583) state it repeatedly
and symmetrically. Every "solves" hit is correctly scoped: "**nearly** solves S₅ (0.921)"
**at trained length T=64** (1463, 1484, 1526), explicitly distinguished from
length-generalization. No bare "delta solves S₅". ✓

**Plus: corrected v0.3 citation in the bpb appendix. LANDED.** §appendix_bpb
"Checkpoints and reproducibility" (2607–2621): x-mode catastrophic (~18 nats/token), y-mode
re-export verified to ≤2×10⁻⁵ nats, "Only the `v0.3` revision carries this corrected
re-export; the earlier public revisions (`v0.1`, `v0.2`) … predate the fix … cite `@v0.3`
only." v0.1/v0.2 flagged exactly as required. ✓

---

## 3. VOICE CONSISTENCY — PASS (register holds top-to-bottom)

Per-section register schema (relative to the §1 reader-first / outside-in exemplar):

| § | Section | Register | Drift? |
|---|---|---|---|
| — | Abstract | reader-first, outside-in (the seed crystal) | anchor |
| 1 | Introduction | reader-first; door-building; codenames arrive late as instances | none |
| 2 | Background | reader-first; **pangenomics re-anchored** — opens "The stakes are the general ones of sequence modeling at scale, and one workload makes them concrete" (459), demoting the terabase workload to an example | none (R1 debt repaired) |
| 3 | Architecture | concrete-mechanism, reader-glossed; each symbol introduced; shape numbers motivated ("not arbitrary", 778) | none |
| 4 | Systems | measured/occupancy register; the R2 epicenter is fully neutralized | none (was R2 epicenter) |
| 5 | LM Results | careful parity register; train-curve explicitly demoted to diagnostic | none |
| 6 | Expressivity | scoped efficiency register; symmetric admissions (delta also length-bounded) | none |
| 7 | Formal | pedagogical "What to take from set X" framing; plain-language-first, Lean id in parens | none |
| 8 | Related Work | **neutral**; GDN-2 "reported by its authors to outperform GDN" (2055), no adversarial verbs, no "concurrent prior art", no timeline alibi | none (R3 competitor-framing cleaned) |
| 9 | Limitations | plain, single-statement limitations; protest register removed | none |
| 10 | Conclusion | abstract-voice restatement; "one artifact shown to exist, not a scaling law" (2267) | none |
| 11 | Predictions | claim+scope+falsifier (the positive exemplar) | none |
| 12 | Future Work | modest; "stronger raw-write baseline" notes it closes the entangled contrast (2405–2420) | none |
| App A | E63→E88 lineage | **reframed as provenance for an outsider** — "internal codenames … carry no meaning beyond their ordering and are used nowhere in the body" (2501–2504) | none (R1 saturation repaired) |
| App B | Held-out bpb | clean methods appendix; corrected-citation home | none |

No section reads in a different register than §1. The body no longer abandons the
abstract's voice.

---

## 4. SEAMS / CONTRADICTIONS — minor only

**S-1 (cross-reference, low-moderate). Line 339: misdirected "(§5)" — should be "(§6)".**
In contribution 2, "raw-write under-reaches the delta update's *S₅* length generalization,
and even at a tuned best-effort budget it does not catch up **(§5)**." The S₅
length-generalization and the "tuned best-effort budget" (to-competence) runs live in **§6**
(Expressivity, 1397–1630); **§5** is LM loss/bpb only. An outside reader following the
pointer lands in the wrong section. The very next clause correctly cites "(§6)". → change
`(§5)` to `(§6)`.

**S-2 (numeric wobble, low). Five vs six orders of magnitude.** §3 (849) says the 131,072
recurrent-state scalars are "about **five** orders of magnitude above the log₂6 ≈ 2.6-bit
floor"; §6 (1219–1220) says an 8 M model "exceeds those floors by … **six** in
recurrent-state scalars per token." Same quantity, two loose "roughly/about" magnitudes;
131,072 / 2.6 ≈ 5×10⁴ ≈ ~5 orders, so §3's "five" is the more accurate figure and §6's
"six" is mildly generous. Both are hand-wavy and non-load-bearing, but the two passes
disagree. → reconcile to "five".

**S-3 (residual register, low). "a defended null" ×2.** Claims-table row 2 label and the
caption (406, 443; and row 4 body) retain "a defended null" — a faint echo of the R3
honesty-protesting register. It is, however, located in the claims table, which is the
*designated* scope-structure home for defense, so it is the least-bad place for it. Borderline;
leave or trim at author discretion.

**Non-issues checked and cleared:**
- BL-1 is stated in §6 (primary, full) and §9 (deferring pointer) — *two places but not
  contradictory*; see reconciliation (b). Not "zero", not a contradiction.
- Floor argument appears in §3 (846–852, as a preview) and §6 (1212–1224, full) — preview +
  home, consistent, both point to §6.
- S₃ control values differ by experiment (8M: GDN 0.72 / M²RNN 0.31; 1.3B fine-tune: GDN
  0.928 / M²RNN 0.748) — *not* a contradiction; the text scopes each to its scale.
- "0.973" in the Common-Pile control (2600) is M²RNN-CMA's Common-Pile bpb, not the train
  bpb — context is explicit (2599–2601); no scope collision.
- All figure/table labels (`tab_claims`, `fig_arch`, `tab_ablation`, `fig_lm_racers`,
  `tab_s5`, `fig_s5_bars`, `fig_hybrid`, `tab_s5_1p3b`, `fig_s5_symmetric`,
  `fig_1p3b_lengthgen`, `tab_bpb_landscape`, `sec:appendix_bpb`) resolve; no dangling
  `@`-reference found.
- §1 roadmap (364–374) matches actual section order §1–§12.

---

## 5. RESIDUAL STANCES + NEW UNCANNINESS — effectively clean

Whole-document grep + read:

- **R2 (production / deployed / released-production / "saturates the GPU" / "full
  utilization"): 0 hits.** The single most cross-flagged defect of the prior audit is fully
  purged, including from the commented-out alternate abstracts.
- **R1 (vantage collapse — "the racer" / "stitched" / "comma-pile" / "on the rack" / bare
  Lean ids / snake_case): 0 hits** of the named insider codenames; E88/Emender defined at
  first use (244–247); Appendix A reframes E63/E88 as provenance and states they "are used
  nowhere in the body" (2503); Lean theorems carry plain-language statements with the
  identifier in parentheses throughout §7.
- **R3 (pre-litigation — "honest null/scope", "not a buried caveat", nulls relitigated):**
  no hits except the mild "defended null" ×2 in the claims table (S-3 above). The §8
  competitor pre-litigation is gone (neutral GDN-2 / M²RNN treatment).
- **R4 (process/QA leaks — "(verbatim)", "audit-recommended wording", meta-credibility,
  cosmic/colloquial/emotional): 0 hits.**
- **New flat/garbled/over-corrected spans introduced by the rewrite:** none found. The
  prose reads as one hand; the §6 symmetric-admission passages ("The delta update is also
  length-bounded") are dense but coherent, not garbled. No "smoothing into new uncanniness".

---

## Bottom line

Ship-blocking issues: **none.** The number/claim integrity gate, all four mandatory
reconciliations, the voice gate, and the residual-stance gate all pass. The only items for
the author are three minor, non-load-bearing touch-ups — **S-1** (`§5`→`§6` at line 339,
the one real seam), **S-2** (five-vs-six orders at lines 849/1219), and optionally **S-3**
("defended null") and the reconciliation-(b) soft note (fully delete vs. keep the §9
deferring pointer). Verdict **RESIDUAL (minor)** rather than CLEAN solely to surface
these spans; the document is otherwise ready.

*READ-ONLY audit. `paper/main.typ` was not modified.*
