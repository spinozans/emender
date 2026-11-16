# GRANULARITY SEPARATOR: chunk-granular decision *generation* is an expressivity weakening

Status: **predictions pre-stated (this revision); sweep pending.** The
predictions below were committed before any separator training run. Results,
receipts and the verdict are appended after the sweep lands; nothing in the
prediction section is edited after the fact.

## 1. Hypothesis and scope

The operator's hypothesis: with state-dependent, irreversible management ops
(no inverses by design), a decision delayed past the information it acts on is
permanent loss; therefore a C-granular decision schedule is a strictly weaker
machine — an expressivity weakening, not an implementation detail. The Lean
theorems (`formal/lean/OPERATOR_ALGEBRA_STATUS.md`, `MemoryAlgebra.Attention`)
cover only *fixed* op sequences: chunked == sequential under a fixed decision
and numerical-gate schedule. Nothing covers decision **generation** at chunk
granularity. This experiment is the empirical witness for the generation-level
gap, at probe scale.

Scope (what the result licenses): the separator shows the **chunked form of
THIS algebra** — the registered five-op matrix store with entry-pinned read
weights and entry-pinned free-slot sets (`schedule.route_chunk`, `read.py`,
`scan.routed_chunk_scan`) — cannot express timely write-then-read inside a
decision interval. It is a claim about a semantic axis of this algebra, not a
general theorem about all chunked attention; §7 states precisely what a Lean
formalization would need to prove and what remains conjectural.

## 2. Why the originally suggested destruction mechanisms do not close

The operator suggested (i) protect-before-overwrite and (ii) timely-merge as
candidate mechanisms. Checked against the actual algebra (registered
thresholds: `tau_match=0.45`, `tau_merge=0.85`, `tau_pop=0.10`), neither is
exactly implementable; the analysis was approved by the operator with the
redesigned task family below (supervisor decision on record):

* **MERGE is geometry-pinned.** A payload `tau_merge=0.85`-similar to two rows
  forces the two rows within cosine `2·0.85²−1 = 0.445` of each other. But
  `tau_match=0.45` means fresh (per-token) routing self-deduplicates: a second
  write 0.445-similar to an existing row CORRECTs it instead of creating the
  near-duplicate row. The mergeable pair the mechanism needs cannot exist under
  the arm that would have to exploit it. This is corroborated empirically: K6's
  per-token MERGE count collapsed to 134 events in 20M targets
  (`receipts/k6_probe.json`) — MERGE is not a load-bearing path in this algebra.
* **Eviction is un-aimable at depth 1 and escapable at depth >= 2.** Layer-1
  payloads are fixed functions of the token embedding; POP/CORRECT victims are
  chosen by cosine against window-random row contents. "Free the RIGHT slot"
  degenerates to a geometry lottery that hits both arms equally, and deeper
  stores give the chunked arm entry-stable escape channels (self-keyed registers
  plus the strength/norm read channel). The overflow-overwrite story is
  therefore not a clean per-token capability anywhere in this architecture.

What *is* exact and strategy-independent in the registered code — proven on
fixtures in `test_granularity_separator.py` — is a pair of channel-level
properties that no payload geometry can escape:

* **P1 (entry-pinned free-slot set).** `route_chunk` enumerates APPEND slots
  from the chunk-**entry** liveness; a slot freed mid-chunk is unusable until
  the next boundary. (Test: `test_freed_slot_not_appendable_same_chunk`.)
* **P2 (entry-pinned read weights).** `read.read_weights` is evaluated against
  the chunk-entry state; a row dead at chunk entry gets attention **exactly 0**
  for every read of the chunk, whatever its content is by read time. The
  in-chunk write-then-read channel does not exist at granularity C — the
  chunked read is *bit-identical* under any perturbation of a mid-chunk
  APPEND's payload. (Test: `test_chunked_read_blindness_is_exact`.)

P2 is the sharpest form of the thesis: *a read decision generated before the
information it must act on is exact, permanent loss for that demand* — the
row is live in the occupancy log, holding the answer, and the generated
schedule still puts exactly 0 weight on it.

## 3. Arms

Both arms are the **same registered eager scan** (`scan.routed_chunk_scan`,
`fused=False`) with the **same registered five-op `RoutingPolicy`** and the
**same probe-scale model** (~70k params: width 64, depth 2, 1 head,
value_dim 32, rank 16, window 16 tokens); only `chunk_tokens` differs:

* **per-token (K6 form):** `chunk_tokens=1` — decisions and reads against the
  CURRENT state at every token. `test_per_token_arm_matches_k6_oracle` holds
  this arm to K6's `sequential_oracle_scan` decision-for-decision (integer
  channels bit-equal; float channels within the fp32-vs-fp64 floor, K6's
  NOISE_FACTOR=8 discipline), so the separator's per-token arm and the K6
  lane are one semantic.
* **chunked (K5B-path dial):** `chunk_tokens ∈ {8, 32, 128}`. The K5B
  whole-scan is a bit-identical host-cost specialisation of this same scan
  (per its receipts), so the eager path here is the same semantics at probe
  scale.

## 4. Task family

**SEP-AR — write-then-read inside the decision interval.** One window
(16 tokens): `BOS, PAD×3, P₀..P₅, K_{π₀} v_{π₀}, K_{π₁} v_{π₁}, K_{π₂} v_{π₂}`,
where `Pᵢ = pair(kᵢ, vᵢ)` is a *combined* (key,value) token (8 keys × 32
values = 256 pair tokens; the combined token is how a fresh binding enters the
store as ONE write whose payload can carry the key-cluster and value
directions — with separate key/value tokens the layer-1 payload could not
represent the window-specific binding, a different failure than the one under
test). The three probes are bare key tokens at positions 10/12/14; the target
is the paired value token (revealed as the next token, teacher-forced).
Accuracy is the value-restricted argmax over the 32 value tokens;
**chance = 1/32 = 0.03125**. Cross-token information flows only through store
reads (every other component is position-wise), so the task's dependence on the
read channel is total.

The pre-registered stratification is the **same-chunk predicate**: a probe is
*blind* iff its pair's write executes inside the probe's own chunk (then the
row is dead at the probe's chunk entry, so its read weight is exactly 0 for
the whole chunk). The predicate table per arm:

| arm | pairs 0–3 (positions 4–7) | pairs 4,5 (positions 8,9) |
|---|---|---|
| C=1  | readable (fresh decisions) | readable |
| C=8  | readable (entry-live at chunk 1) | **blind** (written mid-chunk 1) |
| C=32, C=128 | **blind** (whole window is one chunk, empty entry) | **blind** |

At C=32/128 the window is a single chunk whose entry store is empty: every
read weight in the window is exactly zero and no layer has entry-live rows —
the model is position-wise only, at any depth.

**SEP-EVICT — overflow destruction (corroboration, not the headline).** The
same window at **rank 8**: the store is full through the probe phase
(BOS+PAD+6 pair rows = 8), so the revealed-value tokens and any unclustered
probes face POP/CORRECT/NONE decisions whose victims differ between the
entry-stale argmin (chunked) and the fresh argmin (per-token). Destruction
events (a POP or foreign CORRECT hitting a probed pair's row strictly between
its write and its probe) are counted per run and cross-tabulated with probe
accuracy. Direction pre-stated below; magnitudes honest/best-effort.

## 5. PRE-STATED PREDICTIONS (committed before any training run)

Protocol: per-arm LR sweep {3e-3, 1e-3, 3e-4, 1e-4}, 3 seeds {3401, 3402,
3403}, 3000 steps x batch 256 (minutes per run), fixed eval set (seed 4999,
4096 windows, 12288 probes) shared by every cell. SEP-EVICT: arms {1, 8},
LRs {3e-3, 1e-3, 3e-4}, same seeds, rank 8.

Honest pre-registration disclosure (recorded before the sweep): (a) a harness
smoke run (200 steps, one C=8 cell, far from convergence) confirmed the
plumbing end to end; (b) a one-cell calibration of the PER-TOKEN arm (lr 3e-3,
seed 3401, 3000x256) reached 0.9989 value accuracy, which is what fixes the
P-1 threshold's feasibility at this budget. No chunked-arm cell at protocol
budget has been run at the time of this commit; the chunked predictions
below have not been conditioned on any training result.

* **P-1 (per-token learns).** The per-token arm reaches value accuracy
  >= 0.85 at its best LR (mean over seeds), every seed >= 0.70. This is the
  precondition for the separator to be valid at all: the task must be
  learnable by the fresh-decision form.
* **P-2 (single-chunk chunked at chance).** C=32 and C=128 (the whole window
  is one chunk): value accuracy <= 0.0625 (2× chance) in **every**
  (LR, seed) cell. Mechanically expected: exactly-zero reads at every layer.
* **P-3 (C=8 partial — the middle of the curve, pre-quantified).** Readable
  probes (pairs 0–3): accuracy within 0.15 of the per-token arm's (the
  entry-live rows' contents are unchanged since entry, so the same learned
  geometry serves both arms there). Blind probes (pairs 4,5): accuracy
  <= 0.55 — the bound is the *churn fallback*: the only entry-live channel
  that can carry a mid-chunk pair is a single latest-pair register (one
  CORRECT target per entry-argmax per layer), which answers only when the
  blind probe asks the latest mid-chunk pair (probability 1/2 within blind
  probes), giving ~0.5·(per-token quality) + 0.5·chance + margin. Overall
  C=8 expected in [0.45, 0.85]. The distinct-rows vs collapse geometry
  trade-off (collapsing all pairs into one register caps overall accuracy
  near ~1/6 of per-token quality) keeps both branches well below per-token.
* **P-4 (mechanistic receipts).** For the best C=8 run: the recorded read
  channel puts **exactly 0.0** weight (max over the receipt set) on the blind
  probes' answer rows, while the occupancy log shows those rows live, created
  by APPEND, holding the pair payload and not destroyed before the probe;
  per-token mean weight on the answer row > 0.2. For C=32/128: the read
  channel is identically zero for the whole window.
* **P-5 (SEP-EVICT direction).** Per-token ≈ its SEP-AR accuracy (the
  value-reveal tokens CORRECT harmlessly onto their own pair rows after the
  answer was demanded). Chunked C=8 strictly below its SEP-AR readable-probe
  level (destruction spillover: the entry-stale POP argmin can kill a
  readable pair's row before its probe, and the blocked value-CORRECT escape
  on blind pairs forces POP/NONE there). Destruction events logged per run.
* **Refutation clause (verbatim).** If a chunked arm (C ∈ {8, 32, 128}) learns
  SEP-AR anywhere in the sweep — any LR, any seed, reaching the per-token
  threshold of 0.85 value accuracy on the **blind-probe** subset (for C=32/128
  this is every probe) — that is a genuine refutation of the expressivity claim
  and is reported as such: it reclassifies the granularity effect as
  learnability, which is also a publishable answer.
* **Inconclusive clause.** If the per-token arm never reaches P-1, the probe
  gradient is unresolved at this scale and no expressivity claim is made;
  report inconclusive, tune (LR/steps) within the probe budget, and say so.

## 6. RESULTS (pending the sweep)

To be appended: results table (per arm: best LR, per-seed accuracies, overall
/ readable / blind), the mechanism receipt (`receipts/separator_mechanism.json`),
the SEP-EVICT destruction receipts, and the verdict
(confirmed / refuted / partial / inconclusive) against §5.

## 7. Theorem shape (what a Lean formalization of this claim needs)

The existing results are conditional on a *fixed* decision and gate schedule;
they prove replay (chunked == sequential) and the no-inverses facts
(injection propagation, zero diagonals, state-dependent death non-affinity).
The separator's claim needs a *generation-level* statement, in three parts:

1. **Channel lemma (provable now, machine-checked by
   `test_chunked_read_blindness_is_exact`).** For the chunked router
   `R_C`, the read at any token `t` of chunk `k` is a function of the
   chunk-k entry state, the chunk-k entry liveness, and the payload/query
   lanes of chunk k alone: in particular it is invariant under any change to
   the content of a row first written after the chunk-k entry, and under any
   change to the free-slot set after the chunk-k entry. Formally: the
   chunk-k read map factors through the entry pair `(S_{k·C}, live_{k·C})`.
2. **No-recovery lemma (extends the existing no-inverses facts).** Within
   chunk k, a value whose only presentation precedes its demand cannot reach
   the demand through any op the router can emit: APPENDs into entry-dead
   rows have exactly-zero read weight (lemma 1), APPENDs into mid-chunk
   freed slots are unreachable (P1: the ordinal enumeration is entry-pinned),
   and CORRECT/POP destroy entry-live content irreversibly (injection
   propagation; zero diagonals). Hence if the answer at `t` requires both the
   mid-entry value and all entry-live content, no generated schedule
   satisfies it — a permanent loss, as no inverse op exists to undo the
   overwrite that would make room.
3. **Separation witness (this experiment).** SEP-AR instantiates the premise:
   the probed pair's write and its demand both fall inside one decision
   interval at every C > 1 in the dial, and the demand requires the pair's
   row. The reachable-answer classes at granularity C are then strictly
   contained in the per-token class — an expressivity weakening by
   construction, independent of parameters and training.

What remains conjectural: the strategy-level claim at depth >= 2 (multi-layer
redundancy and self-keyed register escapes exist in principle; they are
bounded here by the empty-entry argument for single-chunk windows and by the
churn-bound analysis of P-3 for C=8, and the sweep's per-probe stratification
is what tests those bounds empirically rather than by proof).

## 8. The two-part signature

The K6/K5B dial gives the **learnability side**: on probe-quality LM loss,
every granularity is solvable and the gradient is monotone and accelerating
into the per-token limit (0.390 nats/byte over the full chunk dial,
`K6_SEQUENTIAL_MATRIX.md`). The separator gives the **expressivity side**: a
task family where the chunked form cannot represent the solution at any
parameters. Together they sign the operator's claim: decision granularity is
both a quality dial and a semantic axis of this algebra — chunk-granular
decision generation is a weaker machine, and the LM-scale loss gap is the
learnability shadow of that weakening.

---

## RESULTS (appended post-sweep by the supervisor session after the lane hit its 6 h ceiling mid-EVICT; every number below is from the lane's own committed-prediction code and per-cell receipts; nothing in the prediction section above was edited)

**SEP-AR, complete: 48 cells (4 chunk sizes x 4 LRs x 3 seeds), best-effort LR sweep per arm, chance = 1/32 = 0.031.**

| chunk | value_acc (all 12 cells) | readable | blind |
|---|---|---|---|
| 1 (per-token, == K6 semantic) | **0.856 – 0.999** | 0.999 | n/a (no blind probes exist) |
| 8 | 0.659 – 0.821 | **0.984 – 0.989** | **0.289 – 0.322** |
| 32 | 0.030 – 0.032 | — | — |
| 128 | 0.028 – 0.033 | — | — |

Verdict against the pre-stated predictions:
- **P-1 (per-token >= 0.85): MET** — 12/12 cells above 0.85; best 0.9989 (the disclosed calibration cell).
- **P-2 (C=32/128 <= 2x chance, mechanically): MET** — every cell of both arms sits AT chance across all four learning rates and three seeds. This is a capability ceiling, not a learning failure: no configuration of the chunked machine reaches even 2x chance.
- **P-3 (C=8 middle of the curve): MET** — readable 0.984-0.989 (within 0.015 of per-token, bound was 0.15); blind 0.289-0.322 (bound was 0.55; the pre-quantified latest-pair churn fallback dominates the blind residual).
- **P-4 (weight exactly 0.0 on live blind answer rows): RECEIPTED** in `receipts/separator_mechanism.json` — the scan's own recorded read channel shows 0.0 attention weight on rows the occupancy log proves live and holding the answer, for every blind probe class, at every layer.
- **P-5 (EVICT direction): PARTIAL** — the per-token evict cells ran (rank-8 overflow: 0.93-0.97, the arm survives destruction events); the chunked evict cells did NOT run (the lane died at its ceiling mid-sweep). Corroboration incomplete; marked honestly. It is not load-bearing for the headline.

**Refutation clause: not triggered.** No chunked cell at any LR or seed reached 0.85 on blind probes. The expressivity reading stands; the learnability reclassification was not forced.

**The claim this receipts:** for the registered five-op matrix algebra with entry-pinned reads and entry-pinned free-slot sets, chunk-granular decision GENERATION is an expressivity weakening. The chunked machine is mechanically blind to any information written inside its decision interval — attention weight exactly 0.0 on the live answer row — while the per-token machine, through the identical code, operators, thresholds and parameters with only the granularity dial moved, reads it at 99.9%. Combined with the K6 probe gradient (2.0278 vs 2.4173 nats/byte on natural text), the two-part signature is complete: the learnability side (real-text quality, accelerating into the per-token limit) and the expressivity side (a task family the chunked form cannot solve at any parameters).

Files: `receipts/separator_ar_*.json` (48 cells), `receipts/separator_ar_sweep.json` (aggregate), `receipts/separator_mechanism.json` (the P-4 zero-weight receipts), `receipts/separator_evict_c1_*.json` (5 cells, partial), `granularity_separator.py` (the lane's final code, uncommitted at its death, included here). The `.pt` checkpoints are NOT committed (program rule).
