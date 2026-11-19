# X46 DESIGN — THE GROWTH SEPARATOR ON A CONSTRUCTED TASK FAMILY (GAS)

Status: **the construction and the derived ceilings are COMMITTED
(`ff9098b5`, machine-checked in `test_x46_growth_separator.py` and
`receipts/x46_selftest.json`) BEFORE any headline arm.** The pre-stated
predictions live in `PREREGISTRATION_X46_GROWTH_SEPARATOR.md` (DRAFT —
operator sign-off REQUIRED before any headline arm; the runner is
fail-closed on the `OPERATOR-SIGNED` mark). Nothing here is edited after
the fact.

## 0. The theorem's missing empirical form

The X41 theory: a fixed-state machine is an automaton over its own state
algebra, and emendation+growth is the only finite-precision escape. The C3
program (`RESULT_GDN2_GROWTH_C3.md`) receipted the NATURAL-TEXT regime:
the registered τ_sat=96 is unreachable (trained PR ~8 of 128 at any feasible
horizon), growth fires only at the calibrated τ_cal≈19.2, the gain is
inside the endpoint band, and grown-large ≈ born-large at matched grown
state. **Natural text does not bind** — the fixed state is never forced to
alias. The separator's whole point is therefore untested: no run in the
program's history has ever put a fixed machine in a regime where its state
provably MUST alias, and asked whether growth is the escape.

The untested form is the CONSTRUCTED TASK FAMILY — the discipline of
`granularity_separator.py` and `key_separation_probe.py`: a registered task
construction, a DERIVED fixed-rank ceiling (from the state algebra, never
assumed from transformer-vs-RNN folklore), pre-stated predictions with
refutation clauses, fail-closed.

## 1. The task family: GAS (growing-associations stream)

One document-bounded constructed stream per window (never Pile material;
model state never carried across windows; constructed documents are the
whole point). The window:

* **Phase A (introduce).** `n` DISTINCT key identities, each presented once
  bound to a value drawn iid from a pool of 128, per-window random.
* **Phase B (probe).** every binding probed exactly once, in a per-window
  shuffled order: the probe names the KEY, the target is the bound VALUE,
  revealed teacher-forced immediately after (the granularity separator's
  reveal convention).

The number of simultaneously-live associations is `n` minus what the
machine has already answered — and all probes come after all introductions,
so the live count at the first probe is exactly `n`, and it stays within
`[n − j]` as probe `j` passes: the live count grows past any fixed capacity
within the stream. This is the dial (`n ∈ {64, 320, 640}`, §4).

**Why values are iid (not unique per window).** The ADDRESS is the key: each
key has exactly one binding per window, so value repeats create no
ambiguity. And the leak analysis is sharp anyway: a probe of an evicted key
cannot find its value by addressing — its row is gone, and any live row
holding the same VALUE is filed under a different key's content, which a
key-addressed query does not match (the machine cannot both retrieve by key
and accidentally match by value). The honest accidental-correctness floor is
chance 1/128, and it enters the derived ceilings explicitly.

**One symbolic construction, two encodings.** The machines have different
input interfaces; the construction is defined once at the symbolic level
(`sample_window`) and encoded per family:

* **MATRIX ENCODING** (the ours arms): a binding is ONE combined pair token
  `PAIR(k, v)` — the granularity separator's writability requirement. At
  the first store layer the payload is a fixed function of the token, so a
  binding presented as separate key and value tokens is unrepresentable at
  depth 1 (the value's write carries no key).  A pair token's EMBEDDING is
  FACTORIZED — `key_table(k) + value_table(v)` — a fixed function of the
  token like the separator's flat pair table, but composed so the key
  component is in the payload by construction and the probe's bare KEY
  token shares the same key row (the flat 82,690-row pair table is
  embedding-starved at probe budget — the pre-sign-off calibration cell
  stalled at ln(128), ~5 gradient touches per pair token at 400 steps; the
  factorized table gives every key and value identity thousands of
  touches — the amendment is disclosed in the RESULT doc §0.5).  The head
  is the value classifier (Linear(width, 128)): the task is "which
  value", chance 1/128 exactly.  A probe is the bare KEY token; the answer
  is the next-token target at the probe position.  Window length `3n+1`.
* **BYTE ENCODING** (the GDN2 arms): a binding is the 3-byte pattern
  `(key_hi, key_lo, value)`; a probe is `(key_hi, key_lo)` and the target is
  the value byte, predicted by the byte LM from the key_lo position (its
  own next-byte form). GDN2's short conv (d_conv=4) makes the multi-byte key
  bindable at write time; the 2-byte key alphabet spans 640 key identities
  (the single-byte alphabet cannot — 256 bytes minus value/marker ranges).
  All four byte ranges are disjoint (`0x02..0x81` values, `0x90..0xA3`
  key_hi, `0xB0..0xCF` key_lo), machine-checked. Window length `6n+1`.

Both encodings decode to the SAME symbolic stream (machine-checked:
identical probe order, identical (key→value) targets).

## 2. The machines (probe scale, the separator's discipline)

* **`ours-fixed`** — the registered store (`MatrixBlock`/`MatrixMixer`,
  imported, never re-written) at probe scale (width 128, depth 2, one head,
  value_dim 64), rank **128** of 2048 slots, growth OFF. Runs the K8
  `GrowingMatrixMixer` with `grow=False`: bit-equal K6 per-token semantics
  at rank ≤ 128.
* **`ours-growable`** — the SAME model, the SAME parameters, the K8
  machinery verbatim (`convert_model_to_growing`): start rank 128, cap
  2048 (snaps {64,128,256,512,1024,2048}), the REGISTERED trigger
  (`scan.repeat_write_rate` at 64-token boundaries, smoothed over 4,
  firing at ≥ 0.5). Rank is capacity, never parameters.
* **`gdn2-fixed`** — a probe-scale GDN2 byte-LM: 2 blocks of the registered
  mixer shape `GDN2ExternalMLPLayer(dim=576, expansion=2, head_dim=128,
  H=2, d_conv=4)` (the committed smoke's block), H=2.
* **`gdn2-growth`** — the same machine, every core wrapped by the committed
  `GrowingGDN2Core` (imported verbatim from `gdn2_growth_c3.py`, the
  registrar/optimizer-surgery/EMA discipline imported with it), heads snap
  {2,4,8}, the CALIBRATED-τ trigger (§5).

Probe scale is the separator's own register (the granularity and
key-separation probes are ~30k-parameter machines; the 150M GDN2 at
probe-scale task budgets is a different experiment — the escalation list
includes the scale ruling). The state ALGEBRA each ceiling is derived from
is scale-free: H·d_k and the slot count are the same at any width.

## 3. The derived ceilings

### 3.1 OURS (rank R): the slot-count bound + the aliasing swap argument

**(i) The slot count is a HARD bound.** The store holds exactly R rows. At
any token the live rows are ≤ R (mechanical). A binding can enter a full
store ONLY by destroying a live row first: the registered five-op chain at
a full store with an unmatched write issues POP (if the worst live
similarity ≤ τ_pop) or NONE — never APPEND — and the NEXT unmatched write
APPENDs into the freed slot. Machine-checked on the registered oracle
(`check_slot_lemma`): live rows ≤ R at every token; every post-fill APPEND
is paid for by a POP; the state carries ≤ R distinct payload directions.
Consequence: at most R bindings are simultaneously addressable, whatever
the learned geometry — no key-separation-style content trick escapes a
count of slots.

**(ii) Aliasing: at most one probe per shared row (the swap argument).** A
row's content is the decayed run-integral of the payloads written to it
(the algebra's payload form, machine-checked in
`test_slot_value_is_decayed_run_integral`): two bindings aliased into one
row give a positive blend `a·f(v_i) + b·f(v_j)`, a, b > 0. The certified
read form is nonnegative (softmax masses × nonnegative strengths), so each
probe's read output is the SAME blend direction up to a positive scalar;
the value-logit gap `logit(v_i) − logit(v_j)` is a fixed linear functional
of that direction and has ONE sign at both probes. The per-window value
swap (probe-A wants v_i, probe-B wants v_j; the swapped window the reverse)
is therefore unsatisfiable: at most one probe of an aliased pair is
answered above the accidental floor. Machine-checked
(`check_aliasing_swap_lemma`, 200 random heads, zero answer both).

**The linear-channel ceiling.** With `L = min(n, R)` addressable live
bindings and the rest at the accidental floor:

```
C_ours(N) = min(1, R/N) + (1 − min(1, R/N)) / 128
```

**Tightness, honestly (the pre-lock calibration's first data point).** The
slot-count ceiling is a valid UPPER bound; the receipts locate the actual
pin, and the first calibration cell pins ours-fixed at 0.2226 at n=320 —
BELOW the 0.4047 bound: slot churn (a POP-freed slot is re-APPENDed before
the evicted binding's probe: ~222 APPENDs + ~95 POPs per window at the pair
writes) and softmax read dilution bind EARLIER than the slot count.  The
bound is not refuted (the clause fires only above); the pin's mechanism
receipts — the recency gradient [0.34, 0.25, 0.18, 0.12] by lag quartile,
the dead stratum at chance — are first-class columns of every cell.

**Scope, honestly (the key-separation precedent).** The bound is exact on
the linear read/routing channels at depth 1. At depth 2 a key-conditioned
NONLINEAR decoder could in principle resolve blends (the residual carries
the probe's key identity; the SwiGLU could correlate it with the blend) —
a conceivable superposition escape, NOT covered by proof. It is covered by
the REFUTATION CLAUSE: a fixed machine beating its ceiling + band REFUTES
the ceiling claim, and the superposition escape (if that is what the
receipts show) is itself a first-class finding — the slot story's boundary
is exactly what this probe exists to locate.

### 3.2 GDN2 (H heads, per-head key dim d_k = 128): the key-space bound

Per head the state is a linear associative memory `S_h = Σ_i c_i k_i v_i^T`
over the head's d_k-dimensional key space (the delta rule adds each new
association along its key direction, decaying the rest). Retrieval for
query q is `S_h^T q`. Three exact statements:

* **The exact regime.** Mutually orthogonal keys retrieve exactly
  (zero crosstalk), so at most d_k = 128 associations per head are
  simultaneously retrievable with NO aliasing (machine-checked:
  orthonormal keys at N=d retrieve with error exactly 0.0).
* **The Welch/frame-potential bound (configuration-free).** For ANY n unit
  key vectors in R^d (n > d), the rms off-diagonal coherence is at least
  `sqrt((n−d)/(d(n−1)))` — the frame potential `Σ⟨k_i,k_j⟩² ≥ n²/d`. At
  N=320, d=128: ≥ 0.0686 (realized 0.0882 on the seed-3401 draw;
  machine-checked across draws).
* **The crosstalk-to-signal ratio has a closed form at the floor:**
  `sqrt(n−1) · rms(cos) = sqrt((n−d)/d)` — **exactly 1 at n = 2d**. At
  N=320, d=128 the floor is 1.22 and the realized value 1.57: every key
  configuration is past reliable per-head retrieval. The per-head pin
  regime begins between N = d (exact) and N = 2d (ratio 1).

**The pre-stated effective capacity: `H · d_k` (the concatenated key
spaces).** Both heads see the same stream; each can in principle hold a
disjoint near-orthogonal key set (128+128 = 256), read out through the
concatenated head outputs. This sum-of-heads form is an OPTIMISTIC bound
(it requires the learned per-head key projections to split the key set
between them) — pre-stated WITH the refutation clause; the per-head exact
bound (128) is the hard floor of the range and the receipts will locate
where the trained machine actually pins.

```
C_gdn2(N) = min(1, H·d_k/N) + (1 − min(1, H·d_k/N)) / 128
```

### 3.3 The ceiling table (chance 1/128, band +0.05)

| machine | capacity | N=64 (control) | N=320 (headline) | N=640 (pressure) |
|---|---:|---:|---:|---:|
| ours-fixed (rank 128) | 128 | 1.0 (no ceiling) | **0.4047** | 0.2063 |
| gdn2-fixed (H=2) | 256 | 1.0 (no ceiling) | **0.8016** | 0.4047 |
| ours-growable grown → 512 | 512 | 1.0 | 1.0 (escape) | **0.8016** |
| gdn2-growth grown → H=4 | 512 | 1.0 | 1.0 (escape) | **0.8016** |

The N ladder does the whole job: 64 is below every ceiling (the
learnability control), 320 is past both fixed ceilings (the headline
separator), and 640 is past the matched grown state 512 (the
capacity-pressure cell where the secondary question lives).

## 4. The two questions

1. **The primary (the theorem's empirical form).** Do the fixed machines
   pin at their derived ceilings while the growable machines escape theirs?
   At N=320: ours-fixed ≤ 0.405+band, gdn2-fixed ≤ 0.802+band, both
   growables ≥ near-1.0 with growth events receipted.
2. **The secondary (the prize).** OURS-growable vs GDN2+growth AT MATCHED
   GROWN STATE (512 slots vs 4×128 key space) UNDER GENUINE CAPACITY
   PRESSURE (N=640 > 512): the first regime where addressable slots vs
   dense superposition can separate when the task FORCES distinct
   addresses. The Wheeler dilution law does not apply here (the task
   forces distinct addresses, not content-similar ones), and the POP
   buffer-vs-memory finding (C2: recycling inert on natural text) predicts
   slots matter exactly when the state must BE the memory. **No
   directional prediction is pre-stated between them** (the task's own
   framing): a tie refutes the slots-earn-their-keep claim under forced
   addressing; an ours win is the first positive cross-family receipt.

## 5. The growth triggers (the escalation's second item)

* **ours-growable: the registered trigger, verbatim, no calibration.**
  The construction IS the write pressure: during Phase-A overflow every
  post-fill write reuses a slot written inside the statistic window, so the
  append-collision rate pins at 1.0 and the trigger fires mechanically at
  the first boundary past the fill. Machine-checked
  (`check_growth_trigger_on_overflow`): rate 0.0 → 1.0 across the fill
  boundary, smoothed mean 0.5 = `grow_trigger`, fire at boundary 128, snap
  64→128. No escalation needed unless the smoke shows otherwise.
* **gdn2-growth: calibrated-OR-forced — ESCALATED, not decided here.** The
  committed GDN2 growth trigger is the per-boundary participation-ratio
  threshold. On natural text the calibrated branch was forced (the
  registered 96 unreachable); on a constructed stream that binds, the
  fixed machine's exhibited PR ceiling is expected far higher, but the
  value is unknown until measured. **Option (a) CALIBRATED (implemented,
  the C3 pattern):** train gdn2-fixed on the constructed stream, commit
  τ_cal = the trained fixed machine's per-layer smoothed-PR ceiling (the
  calibrated-below-registered discipline: growth fires exactly when a
  boundary is MORE saturated than anything the fixed machine exhibits),
  then run the growth arm at τ_cal. **Option (b) FORCED (task-derived,
  NOT implemented):** fire the head-doubling at the first statistic
  boundary past the per-head exact bound (the derived aliasing point),
  receipted as a construction-derived schedule. The runner implements (a)
  and REFUSES (b) by name pending the operator's ruling; the smoke's
  provisional τ=20 is plumbing-only.

## 6. The protocol (the probe conventions)

* **Matched exposure:** every arm sees `steps × batch × n` probe targets
  (identical steps/batch across machines; the byte encoding's longer
  windows are GDN2's own interface cost, disclosed per-arm).
* **Optimization:** AdamW (0.9/0.95, wd 0.01, clip 1.0 — the separator
  convention), one family for all arms; constant LR for 90% of the steps
  then a linear cooldown over the last 10% (the corrected-protocol doctrine
  adapted to the accuracy probe).
* **Scoring (the PRIMARY):** final value-restricted probe accuracy on the
  FIXED eval set (seed 4999), chance 1/128; per-probe lag quartiles and
  the live/dead-at-probe stratification (the ours arms' recorded
  schedules) as the mechanistic receipts.
* **Seeds:** LR sweep on seed 3401; the headline cells confirmed on seed
  3402 (the task's "seeds 3401/3402 on the headline").
* **Band:** 0.05 accuracy (the endpoint convention adapted; pre-stated).
* **Honest automata-bound arithmetic:** every arm's receipt carries its
  `ceiling_record` (capacity, ceiling, ceiling+band, the algebra string).

## 7. What this probe does NOT establish (scope, stated)

* Probe scale; both encodings' learnability below the ceilings is gated by
  the N=64 control and the inconclusive clause (if the growables cannot
  learn N=64, nothing is claimed).
* The derived ceilings' linear-channel scope: the depth-2 superposition
  escape is disclosed and refutation-clause-guarded, not proof-guarded.
* The GDN2 sum-of-heads capacity (256) is optimistic-by-construction; the
  receipts locate the actual pin.
* A win/loss between the two growable machines at matched grown state is
  NOT predicted; both outcomes are pre-stated findings.
* Nothing here is a board or LM claim; the constructed family binds by
  design — that is the point.
