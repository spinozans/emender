# PREREGISTRATION X46 — THE GROWTH SEPARATOR ON GAS (the constructed task family)

**Status: DRAFT — PENDING OPERATOR SIGN-OFF.** The task construction and
the derived ceilings are committed (`ff9098b5`, machine-checked:
`test_x46_growth_separator.py`, `receipts/x46_selftest.json`) BEFORE this
registration; the arm runner is FAIL-CLOSED until this file carries the
string `OPERATOR-SIGNED` with a date. The escalation items are in §8 — the
operator's rulings there EDIT this registration before sign-off; the
predictions in §1-§7 are then frozen.

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

## 8. ESCALATION ITEMS (the operator's rulings EDIT this section, then sign)

1. **The derived ceilings' exact algebra** (§4): the ours linear-channel
   ceiling `min(1,R/N) + rest/128` (slot-count + swap argument, depth-1
   linear scope, the depth-2 superposition escape refutation-clause-
   guarded) and the GDN2 key-space ceiling (per-head exact 128;
   sum-of-heads 256 pre-stated as the OPTIMISTIC effective capacity;
   Welch closed form `sqrt((n−d)/d)` pinning per-head reliability at
   n≈2d). Sign off, amend, or restate.
2. **The gdn2-growth trigger semantics on the constructed stream:**
   (a) CALIBRATED (implemented; the C3 discipline: τ_cal = the trained
   fixed machine's per-layer smoothed-PR ceiling on THIS stream,
   `calibrate` commits it before the arms) vs (b) FORCED (task-derived:
   fire at the first boundary past the per-head exact bound — NOT
   implemented, refused by name). The task text permits either
   ("calibrated-OR-forced"); a ruling is required before the gdn2-growth
   headline cells run. The implemented default is (a).
3. **Machine scale:** the GDN2 arms at probe scale (2 blocks of the
   registered mixer shape, per the separator discipline) vs the 150M
   registered machine on the same constructed stream (the C3 lane's
   builder). Probe scale is registered here; the 150M variant is a
   disclosed alternative requiring its own budget ruling.
4. **The n ladder and the headline N=320** (past both fixed ceilings with
   margin; 640 for the matched-state pressure cell): sign off or amend
   before the arms run.
5. **Unexpected ceiling behavior** (post-sign-off, standing): any machine
   pinning well ABOVE or BELOW its derived ceiling + band, any trigger
   that fails to fire on the constructed stream, any growth event
   pathology — escalated with receipts, never tuned live.
