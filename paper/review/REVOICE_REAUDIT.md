# REVOICE PHASE 3 — Automated Re-audit (the review gate)

**Subject:** Garrison, *Emending Nonlinear Recurrence* (`paper/main.typ`), **revoiced** state at commit `6d1215b`.
**Mode:** READ-ONLY review gate. `paper/main.typ` was NOT edited. This document verifies that the Phase-2 revoice (`revoice-2-stances`, diff `4121b57..6d1215b`, +150/−163) removed the three stances + R4 plane-slips **without introducing new uncanniness** (the smoothing risk).
**Frame:** R1 vantage · R2 ontology · R3 posture · R4 plane/register · R5 instrument integrity (definitions in `paper/review/texture_audit/SYNTHESIS.md`).
**Method:** deterministic grep sweep over every original-audit token; full re-read of every changed hunk by 7 per-section readers; two cross-cutting whole-read passes (compounding intensity, prosody/cadence); manual corroboration of §1 opener, §7 process leaks, §8 competitor framing, `tab_claims` R5 contract, and all preserved numerics; `bash paper/build.sh` → exit 0.

---

## VERDICT: **RESIDUAL**

The revoice **succeeded**. Every deep stance source the original audit named as the engine of the "uncanny and fucked up" reaction is **gone from the reader-facing body**, and the smoothing risk did **not** materialize: no broken transitions, no dropped or altered numbers, no unscoped claims, no garbled over-corrections, and no new term hammered into dread. The verdict is **RESIDUAL** rather than **CLEAN** only because a short tail of **MESSY-tier** stance tokens survived the scrub — two of them in conspicuous section headings — and the revoice introduced **one** flat redundant sentence. It is **not REGRESSED**: zero new UNCANNY-tier spans, and the only regression is one MESSY redundancy.

**Severity ledger:** UNCANNY survivors = **0**. MESSY residuals = **~10 spans** (several pre-existing, untouched by the revoice). New regressions = **1** (MESSY). Numerics intact (0.973 / 0.966 / 0.961 / 22,200 / 15.7% / 7,492 / 8,248 / 0.7918 / 0.143 / 1.273 B / H=370 all unchanged). Build clean.

This is a ship-with-optional-cleanup outcome — far closer to CLEAN than to REGRESSED.

---

## 1. Residual-stance check (R1/R2/R3/R4) — survivors

### 1a. CONFIRMED REMOVED (the major wins)

| original defect | rule | status in revoiced text |
|---|---|---|
| "production / deployed / released production weights / production stack/instance/architecture" (11+ sites; the single most cross-flagged token) | R2 | **GONE.** E88 reframed as a result: "the instance reported here" (L164); "released v0.3 weights, loaded strictly" (L1525); "1.3 B stack" (L1133); "1.3 B architecture" (L2087). |
| §1 opener "E88, the 1.273 B-class **production instance** of the Emender…" (canonical #1 specimen) | R1+R2 | **FIXED.** Rewritten outside-in: "A purely nonlinear recurrent language model can be trained to billion-parameter scale… The instance reported here, E88 (1.273 B parameters), reaches 0.973 bits per byte" (L162-164). |
| "the racer" (7+), "stitched" (6×), "comma-pile", "on the rack", "the campaign", self-naming footnote | R1 | **GONE.** → "comparison" / removed / "Common Pile" / "evaluated here". 0 hits each. |
| "the architecture emended its own name"; "independently voted for the architecture's central thesis" | R2 | **GONE** (footnote + flourish deleted). |
| entire "honest"/honesty-protesting register; "not a buried caveat"; "a critic can always ask" (×2); "This is the culmination"; "Honest mirror/null" headings | R3 | **GONE.** `honest`/`honesty` = **0 hits**. Headings renamed to neutral descriptors; critic-questions replaced with explicit "its limitation is…" scope. |
| "(verbatim)" heading; "audit-recommended wording from the formalization gap analysis"; body file paths; "moat"; "holds comfortably"; "frustration"; "substrate of the actual world" | R4 | **GONE.** "moat"→"boundary" (L1957); "holds comfortably"→"holds with margin" (L1797); file paths removed from body. |
| §8 competitor-clearing: "head-to-head", "concurrent prior art", "every incentive", "violate the no-hybrid", GDN-2 submission-date alibi | R3 | **GONE.** §8 now reads as balanced scholarly related-work; GDN-2 is "concurrent with this work… left to future work" (L1891). |

### 1b. SURVIVORS (all MESSY; register schema below)

| # | line | rule | span (verbatim, abridged) | why it survives | touched by revoice? |
|---|---|---|---|---|---|
| S1 | **2003** | R4(c) | "The Emender **retains the lead** under length extrapolation" | Race metaphor — the original audit's **CRITICAL s9#8** finding. All neighboring "racer" terms were neutralized; this race-positioning verb was missed. Reads near-factual but personifies the result as a runner holding position. **Most conspicuous survivor.** | no |
| S2 | **921** | R3 | heading "Per-architecture CMA-ES protocol **(fairness anchor)**" | "fairness anchor" is named verbatim in the R3 violation list. Body rationale (L939) was revoiced but the defensive heading parenthetical was left. Conspicuous (a heading). | body yes / heading no |
| S3 | **1074, 1076** | R3 | heading "The loss tie is FLOP-locked, **not seed-luck**" + "The tie is **not an accident of one fortunate run**." | Honesty-protest-by-negation; both phrases named in the SYNTHESIS list. The block below is properly restructured (Finding/Caveat/Scope) but the heading + opener still lead with defensive negation against an absent skeptic. | no |
| S4 | **939** | R3 | "adopted **to avoid the undisclosed HPO budgets common in nearby work**" | The "frustration/doctrine" affect was removed, but the residual rationale still argues against unnamed nearby papers. Softened but present. | partially |
| S5 | **2432, 2455, 2458-59, 2465, 2470-72** | R3 | "landscape, **not a ranking** … **not an architecture ranking** … **must not be read as a quality verdict** … **not a quality knock**" | The "do not read order as a quality verdict" point relitigated ~5× across heading + caption. Appendix-demarcated → MESSY. (Original logged 4×; comparable.) | no |
| S6 | **274, 277, 278** | R1 | bare Lean ids `emender_m2rnn_one_step_resource_separation_embeds`, `…k_step_separation`, `…flop_class_equiv` in §1 contribution 2 | Contribution 1's analogous slug **was** scrubbed → "Lean-certified; see §4 and §7", but the very next bullet's three slugs were left, making the partial fix internally inconsistent. Demarcated as parentheticals → routable. | adjacent bullet only |
| S7 | **320-328** | R4(a) | reader-facing `RELEASE_V02_PUBLIC_RELEASE_HUB.md` + `…/tree/v0.3` URLs in §1 body | Version strings / repo URLs in the intro body rather than a demarcated availability appendix. Pre-existing; a legitimate release pointer. | no |
| S8 | **1111, 2179** | R3 | "where the architectures **genuinely** come apart"; "tasks requiring **genuine** state-tracking" | Mild honesty-protesting intensifier; here descriptive of the architectures (not author candor). | no |
| S9 | **753-756** | R3 | "where non-solvability is **not** the obstruction" restated in two adjacent sentences | Mild relitigation padding one scoped claim. Pre-existing. | no |
| S10 | **1696** | R3 | "the stronger inseparability claim a reviewer would naturally want; the trusted Lean core does not contain it" | Orients a non-claim around an absent reviewer's wants — but sits inside the sanctioned "What is not proved" block, the legitimate home for scope. | no |

**Source-hygiene note (not reader-facing, no grade):** the figure label identifier `<fig_lm_racers>` (L1015, referenced L2014/2078) still reads "racers"; the rendered prose is clean. A cross-ref anchor, invisible to the reader — flagged for tidiness only.

---

## 2. New-uncanniness / regression check (the smoothing risk) — EXPLICITLY DONE

Every changed hunk in `4121b57..6d1215b` was re-read for: flat/garbled over-correction, broken transitions, an accidentally changed/dropped number or term, and a deleted defensive point that left a downstream claim unscoped.

**Result: ONE regression, MESSY-tier.**

| line | kind | span | why |
|---|---|---|---|
| **1907-1908** | flat / redundant | "M²RNN establishes that nonlinear matrix-state recurrence can be trained at scale in hybrid form." | The revoice replaced the R3 phrase "This is concurrent prior art for the claim that…" with a sentence that **near-verbatim restates the immediately preceding sentence** (L1903-1906 already states M²RNN "demonstrates that it trains at 7 B MoE scale in hybrid form"). Flat redundant filler between the citation and the M²RNN-CMA definition. Not floor-shifting → MESSY. |

**Everything else the smoothing risk could have broken held:**
- **Transitions / antecedents:** all deleted-stance splices close cleanly — the merged §1 obstruction sentence, the ParaRNN "capable team" cut, the "not a buried caveat" cut (L1094), the "a critic can always ask" rewrites, the renamed §6 headings (cross-refs "see *…* below/above" still resolve). No dangling connectives introduced.
- **Numbers/terms:** no drift. Verb swaps "wins"→"exceeds", "edges"→"is ahead of", "wins on state tracking"→"leads" changed **no** numeric values (1.00 vs 0.86; 0.921/0.658/0.117; 0.143/0.090; 0.79/0.36/0.22; etc. all preserved).
- **Scope:** the deleted critic-questions in §6 were **replaced** with explicit "its limitation is…" scope statements — caveats preserved, not orphaned. No claim left unscoped.
- **Bonus fixes (R5 improvements the revoice made):** `tab_claims` rows 2 & 4 now hold **real claims** (the disowned "*No.*"/"*Not claimed.*" propositions neutralized into clean claim/scope — original R5 contract fault **fixed**); and the figure caption "three load-bearing design choices" vs §3's "four" mismatch is **fixed** → caption now reads "§3 enumerates the four load-bearing ingredients" (original 3-vs-4 R5 fault **resolved**).

---

## 3. Compounding + cadence axes (the two the original audit under-measured)

### 3a. Compounding intensity — **RESIDUAL** (no longer dread-ful)
Whole-read recurrence scan, corroborated against the diff:

| term | approx count | dread-ful now? | comment |
|---|---|---|---|
| production / racer / stitched / deployed | **0** in body (was 11+ / 7+) | no | The compounding-dread engines are **purged**. |
| "the Emender" | ~62 | no | The architecture's proper name; recurrence expected for a named subject — reads as a noun, not an incantation. |
| "tie" / "null" | ~20 | no | Domain vocabulary naming the literal §5 result; the argumentative table cells were even neutralized. |
| "ahead" / "leads" | ~13 | no | Permitted factual comparatives on distinct claims; "wins/edges"→"leads/ahead" is a register *improvement*. |
| "released v0.3 checkpoint" / `@v0.3` | ~10 | no | Concentrated in §5 captions / reproducibility pointers where a version string is the legitimate artifact handle. |
| "trusted core" + sorry/admit/axiom list | ~6 / ×3 | no | Literal description of the verified Lean import closure. |
| "honest" / "genuinely" | honest **0**; genuine(ly) 3 | no | Core R3 word purged; 3 "genuine(ly)" describe the architectures, not author candor. |

No replacement word is hammered into new dread → **REGRESSED ruled out** on this axis. RESIDUAL (not CLEAN) because S2 ("fairness anchor") and S3 ("not seed-luck") survive in **headings**, where named R3 tokens are maximally conspicuous.

### 3b. Prosody / cadence — **RESIDUAL** (structurally sound; not regressed)
The revoice did **not** break rhythm or create garbled non-sentences; the deleted-stance splices read cleanly. The relentless hedge/clause-stacked cadence the original audit named persists, **untouched**, in two pre-existing hotspots:

| line | issue | note |
|---|---|---|
| 1097-1100 | hedge-stacked | §5 single-seed caveat: semicolon + nested parenthetical carrying its own relative clause ("the seed noise it would purport to bound") + tacked-on "and we do not." Pre-existing; loses the reader before the period. |
| 1456-1460 | run-on / dangling participle | §6 "…extending its own flattening late-training rate, even doubling the budget *again* would add only ≈0.03…" — comma-spliced run-on, "extending" lacks a grammatical subject. Pre-existing (not in the revoice diff). |

Both are **pre-existing** prose faults, not revoice damage → cadence is **not REGRESSED**.

---

## 4. Register schema (per surviving span)

```
schema: { id, line, rule, severity, origin, disposition }
  rule       ∈ {R1,R2,R3,R4a,R4b,R4c,R5}
  severity   ∈ {UNCANNY, MESSY}          # UNCANNY survivors this pass: 0
  origin     ∈ {survived-revoice, introduced-by-revoice, pre-existing-untouched}
  disposition∈ {ship-as-is, optional-cleanup}
```

| id | line | rule | severity | origin | disposition |
|----|------|------|----------|--------|-------------|
| S1 | 2003 | R4c | MESSY | survived-revoice | optional-cleanup (highest priority: orig. CRITICAL flag) |
| S2 | 921 | R3 | MESSY | survived-revoice (heading) | optional-cleanup |
| S3 | 1074,1076 | R3 | MESSY | survived-revoice (heading) | optional-cleanup |
| S4 | 939 | R3 | MESSY | partially-revoiced | optional-cleanup |
| S5 | 2432+ | R3 | MESSY | pre-existing-untouched | ship-as-is (appendix) |
| S6 | 274,277,278 | R1 | MESSY | survived-revoice | optional-cleanup (consistency) |
| S7 | 320-328 | R4a | MESSY | pre-existing-untouched | ship-as-is |
| S8 | 1111,2179 | R3 | MESSY | survived-revoice | ship-as-is |
| S9 | 753-756 | R3 | MESSY | pre-existing-untouched | ship-as-is |
| S10 | 1696 | R3 | MESSY | pre-existing-untouched | ship-as-is |
| R-1 | 1907-1908 | R3→flat | MESSY | **introduced-by-revoice** | optional-cleanup |

**Suggested (no edits made — this is the READ-ONLY gate):** if a Phase-4 polish runs, the four highest-value touches are S1 (`retains the lead` → "stays ahead at every length" / report the figures), S2 (drop "(fairness anchor)"), S3 (retitle "The loss tie is FLOP-locked" / "…reproduces under matched compute"), and R-1 (delete the redundant L1907-1908 sentence). None blocks ship.

---

*End of re-audit. `paper/main.typ` not modified. Verdict: **RESIDUAL** — revoice verified successful; zero UNCANNY survivors, one MESSY regression, a short MESSY stance tail (two in headings). No edits recommended as a gate; the listed touches are optional polish.*
