# PREREGISTRATION X46 — THE GROWTH SEPARATOR ON GAS (the constructed task family)

**Status: DRAFT — PENDING OPERATOR SIGN-OFF.** The task construction and
the derived ceilings are committed (`ff9098b5`, machine-checked:
`test_x46_growth_separator.py`, `receipts/x46_selftest.json`) BEFORE this
registration; the arm runner is FAIL-CLOSED until this file carries the
string `OPERATOR-SIGNED` with a date. The supervisor's seven rulings
(2026-11-30) are in §8 and are AMENDED INTO this registration (§5, §6 P-2);
the signature gate is the OPERATOR's. On their signature the sweep runs:
`calibrate` (GDN2 τ) → LR grid seed 3401 at n∈{64, 320} → headline seeds
3401/3402 → the n=640 matched-grown-state cell → `assemble`.

## 1. The question

The X41 theorem's empirical form, never yet run: **does a fixed-state
machine pin at its own derived ceiling on a task that provably forces the
state to alias, and does growth escape it?** Secondary (the prize, no
directional prediction): **addressable slots vs dense superposition at
MATCHED GROWN STATE under genuine capacity pressure** — ours-growable at
grown rank 512 vs gdn2-growth at grown H=4 (4×128 = 512) at N=640.

## 2. The registered task construction (GAS)

`x46_growth_separator.py`, committed before this registration: a
document-bounded growing-associations stream — `n` distinct keys bound to
iid values (pool 128), all introduced then all probed in per-window
shuffled order (lag ~n); one symbolic construction, two encodings (combined
pair tokens for the ours machines — the depth-1 writability requirement;
2-byte-key + value-byte patterns for GDN2 — the byte LM's interface). The
`n` dial: **{64, 320, 640}**. Chance = 1/128. Windows are constructed
documents; no Pile material; no state carried across windows.

## 3. The arms

`ours-fixed` (registered store, rank 128/2048, growth off, K6 semantics),
`ours-growable` (the K8 machinery verbatim: start 128, cap 2048, the
REGISTERED append-collision trigger), `gdn2-fixed` (probe-scale GDN2 byte
LM: 2 blocks of the registered mixer shape, H=2, d_k=128), `gdn2-growth`
(the committed `GrowingGDN2Core` wrapper verbatim, heads snap {2,4,8}, the
§8-ruling trigger). Probe scale throughout; the state algebra the ceilings
derive from is scale-free.

## 4. The derived ceilings (committed, machine-checked)

* **OURS, rank R=128:** the slot-count bound (≤ R live rows; a post-fill
  binding enters only by destroying a live row — machine-checked on the
  registered oracle) + the aliasing swap argument (an aliased row's read
  outputs are one shared direction up to a positive scalar; no fixed
  linear head answers both probes of the swap — machine-checked).
  `C(N) = min(1, 128/N) + (1 − min(1, 128/N))/128`.
* **GDN2, H=2, d_k=128:** the linear-associative-memory key-space bound
  (exact retrieval needs orthogonal keys: ≤ 128/head — machine-checked;
  the Welch/frame-potential bound `rms(cos) ≥ sqrt((n−d)/(d(n−1)))` holds
  for EVERY key configuration; the crosstalk-to-signal ratio at the floor
  has the closed form `sqrt((n−d)/d)`, EXACTLY 1 at n=2d, 1.22 at N=320 —
  machine-checked). The pre-stated effective capacity is the
  sum-of-heads 256, optimistic by construction, refutation-clause-guarded.
  `C(N) = min(1, 256/N) + (1 − min(1, 256/N))/128`.
* **The table** (band +0.05):

| machine | N=64 | N=320 | N=640 |
|---|---:|---:|---:|
| ours-fixed | no ceiling | **≤ 0.4047** | ≤ 0.2063 |
| gdn2-fixed | no ceiling | **≤ 0.8016** | ≤ 0.4047 |
| ours-growable (grown 512) | — | escape (≥ near-1) | ≤ 0.8016 + its own ceiling |
| gdn2-growth (grown H=4) | — | escape (≥ near-1) | ≤ 0.8016 + its own ceiling |

## 5. The protocol

Matched exposure (`steps × batch × n` probe targets on every arm); AdamW
(0.9/0.95, wd 0.01, clip 1.0), one family for all arms; constant LR 90% +
linear cooldown 10%; PRIMARY = final value-restricted probe accuracy on the
fixed seed-4999 eval set; LR sweep {3e-3, 1e-3, 3e-4, 1e-4} on seed 3401;
headline cells confirmed on seed **3402**; endpoint band **0.05**
(accuracy); the honest automata-bound arithmetic (`ceiling_record`) in
every arm's receipt; mechanistic receipts (growth events, the ours arms'
live/dead-at-probe stratification, the lag quartiles, GDN2's per-boundary
PR at the calibration and the fires).

## 6. PRE-STATED PREDICTIONS (frozen at sign-off)

* **P-0 (the learnability precondition).** At N=64 (below every ceiling)
  every machine reaches ≥ 0.90 at its best LR (mean over its headline
  seeds). If a machine cannot, its rows are reported INCONCLUSIVE for that
  machine (the key-separation inconclusive clause); no ceiling claim is
  made against a machine that cannot learn the task below its ceiling.
* **P-1 (the fixed machines pin — the primary).** At N=320: ours-fixed's
  best headline cell ≤ 0.4047 + 0.05; gdn2-fixed's best headline cell ≤
  0.8016 + 0.05. **Refutation clause:** a fixed machine beating its
  ceiling + band at any LR or seed REFUTES its ceiling claim and is
  reported as such — with the depth-2 superposition escape (a
  key-conditioned nonlinear decoder resolving aliased blends) as the
  first candidate mechanism to receipt, a first-class finding either way.
* **P-2 (the growables escape).** At N=320 both growable machines beat
  their fixed twins by MORE than the band AND reach ≥ 0.90, with growth
  events receipted (ours: the registered trigger's fires and the grown
  rank; GDN2: the wrapper's events and the grown head counts). A growable
  that ties its fixed twin, or whose trigger never fires, is the declared
  informative null for that machine, reported as such.
* **P-3 (the matched-grown-state prize, NO directional prediction).** At
  N=640 with both growables at grown state 512 (ours rank 512, GDN2 H=4):
  a TIE (within the band) REFUTES the slots-earn-their-keep claim under
  forced addressing; an ours win beyond the band is the first positive
  cross-family receipt (addressable slots beat dense superposition exactly
  when the state must BE the memory); a GDN2 win beyond the band receipts
  the opposite. All three outcomes are pre-stated findings; none is
  preferred.
* **P-4 (the growth mechanics).** ours-growable's registered trigger
  fires in-window on the constructed stream (machine-checked at the
  overflow: rate 0.0 → 1.0 across the fill boundary, fire at 128, snap
  64→128; at the headline shapes 128→256→…), with the grown rank ≥ n at
  N=320 by the escape; gdn2-growth's calibrated trigger fires with the
  per-event PR receipts, and its τ_cal is committed BEFORE its arms run
  (§8's ruling).
* **P-5 (the mechanistic stratification).** On the ours arms at N=320 the
  live-at-probe fraction is ≤ rank/n + ε (the slot-count bound made
  countable), the live-stratum accuracy dominates the dead-stratum
  accuracy, and the lag quartiles show the recency gradient the eviction
  policy predicts (later-probed bindings answered no better than
  earlier-probed ones once evicted).

## 7. Reporting discipline

Results land in `RESULT_X46_GROWTH_SEPARATOR.md` per part as they arrive;
the pre-stated sections of this registration are never edited after
sign-off; every adjudication quotes the ceiling table and the band; a
refutation is reported as a refutation, never excused.

## 8. THE RULINGS (supervisor, all seven -- 2026-11-30; these AMEND this
registration; the OPERATOR-SIGNED gate is the operator's, not the
supervisor's -- the arms run only on their signature)

1. **The derived ceilings' algebra: SIGNED OFF** (the Welch-bound derivation
   for GDN2 and the slot-count+aliasing-swap derivation for ours,
   machine-checked, refutation-clause-guarded on both superposition
   escapes).  No amendment.
2. **The not-tight ours ceiling: DISCLOSED AS-IS, accepted.**  A valid-but-
   not-tight bound with the pin at 0.2226 and the mechanism receipts (slot
   churn, read dilution, the recency gradient, the dead stratum at chance)
   is richer than a tight bound; the refutation clause stays one-sided.
3. **The gdn2-growth trigger: CALIBRATED** (the C3 discipline on THIS
   stream; tau_cal committed by `calibrate` BEFORE the headline arms; every
   fire receipted with its PR value and window position per the C3
   convention).  FORCED stays refused by name; an under-firing calibrated
   trigger in a headline cell is escalated with the receipt, never
   silently substituted.
4. **Machine scale: PROBE-SCALE** (2 blocks of the registered mixer shape,
   the separator tradition -- granularity, key-separation, Wheeler all ran
   probe-scale).  The 150M-machine cell is the disclosed alternative, NOT
   run -- scope discipline; a follow-up can take it to 150M under its own
   sign-off.
5. **The n ladder: {64, 320, 640}** as proposed -- 64 the learnable
   control, 320 the headline (carrying the 2.6x pre-lock evidence), 640
   the matched-grown-state pressure cell where ours-vs-GDN2 at matched
   grown state is the prize read.
6. **The bootstrap: the UNIFORM CURRICULUM LADDER is PRIMARY** (receipted:
   collapses the cold-start saddle, 0.9647 at n=64 in 19 s, identical on
   every arm, optimizer state carried -- fair by construction); longer
   cold cells only as labeled secondary robustness if budget remains; the
   saddle disclosure and the non-fused-AdamW receipt are in the protocol
   section verbatim.
7. **The P-2 escape bar: AMENDED**, with both readings kept and honestly
   separated (the P-2 text above): the mechanism-based escape claim (beats
   the fixed twin beyond the band + events + flat lag gradient) is the
   PRIMARY and was lockable BEFORE any headline cell (the pre-lock cells
   are calibration per the key-separation §6 precedent; this amendment
   would have been illegitimate after the lock); the >= 0.90 solvability
   level is the non-gating SECONDARY, budgeted per the protocol section.
