# X28E — the budgeted-growth tape head: registration DRAFT (pre-commit, pre-launch)

**Status: DRAFT. Uncommitted. No training authorized, no smokes run, nothing
launched.** Commit-and-push of this registration (SHA-bound with code, gates,
generators and `registration.json`) precedes any X28E optimizer update, exactly
the X25/X27 commit-before-launch ancestry. Zero X28E updates exist at draft time.
This draft operationalizes the operator's central hypothesis — **STATES GROW TO
THE SIZE THE TASK DEMANDS, EVEN OUTSIDE THE TRAINING REGIME** — as the
"unbounded fork with reduction": growth AND architectural shrinkage, on the
Lean-certified `a^k b^k` ladder where every bounded-state channel is provably
insufficient. Theory legs: `theory_x28t` (constructive tape model, delivered)
and `StreamingExpressivity.balanced_needs_unbounded_state` /
`context_free_discrimination_impossible` (X21T both-impossible theorems,
activation-free, dimension-free, budget-free).

## The head spec (frozen at commit; single spelling: "slot-store tape head")

Per (layer, head) the recurrent state is a variable-length store of slots; each
slot is a **(key, value) pair in fp32 cells** plus architectural bookkeeping
(live/dead flag, age, usage). All slot arithmetic is fp32.

- **Content-addressed read**: a learned query scores every live slot; the read
  output is the (soft-)matched value mixture. Queries, keys, values are all
  learned.
- **Delta-rule error-correcting write — the DEFAULT op**: the write addresses
  the best-matching live slot and applies the delta-rule correction
  (error-correcting overwrite of the matched slot's value). This is the
  receipted best-igniting op from the memory-head line (stage-1 associative
  recall is where its ignition is receipted; X27 GROWTH_REPORT named this
  design dependency).
- **Soft append gate (grow) — APPEND WHEN THERE IS NOTHING TO CORRECT (T9
  resolution, operator-delegated)**: the gate's input includes the write-error
  magnitude against the best-matching live slot. Zero error (identical
  content, nothing to correct) → **append a new slot**; error present → the
  delta-rule correction fires instead. Identical things accumulate,
  conflicting things reconcile: distinct memories are never destroyed by
  identical writes. The LADDER MECHANISM PREDICTION (registered, receipted
  by routing traces and occupancy — NOT an architectural guarantee; see the
  routing rule's ladder-honesty block): on `a^k b^k`, every `a` requests an
  append and every `b` pops one `a`-slot. The gate strength remains a
  learned scalar around this architectural rule. The ARCHITECTURAL guarantee
  is only the routing implication: an append request allocates only when
  capacity remains; otherwise it merges (R-CAP).
- **POP BY CONTENT (reduction #1 — the true stack op)**: a read carries a
  learned pop strength; the best-matching live slot is marked dead with that
  strength (task-driven shrinkage). The per-`b` popping behavior on `a^k b^k`
  is part of the registered MECHANISM PREDICTION above — receipted by
  traces, not architecturally guaranteed.
- **USAGE-DECAY (reduction #2 — the cache policy)**: per-slot age/usage
  counters with an architecturally fixed decaying strength; slots fall below
  the registered free threshold and are freed. Decay rate and free threshold
  are FIXED ARCHITECTURAL CONSTANTS, bound in `registration.json` at commit,
  never learned, never tuned post-hoc. **T2 resolution (operator-delegated):
  the decay constants are registered so that a live slot's fade horizon is at
  least 10x the longest tested word** — decay is provably inert within the
  ladder (calculation receipted at commit) and cannot be the binding
  constraint on any arm; it exists for production-length garbage collection.
  A deep-k failure therefore cannot be attributed to cache policy.
- **MERGE UNDER BUDGET (reduction #3 — graceful degradation)**: when a budget
  cap B is hit, append pressure degrades into a delta-rule merge onto the best
  match (no new slot is ever allocated past B). At budget the head therefore
  becomes a bounded slot store using delta-rule corrections — GDN2-LIKE as an
  analogy, not an equivalence (the native fixed-matrix recurrence is not
  literally instantiated, and pop/decay remain present); this degeneration is
  a feature, not a failure mode.

**Zero learned management.** All three reduction ops are architectural. The
gradient touches ONLY keys, values, queries, write/pop/append strengths and
gates. No learned allocation policy, no learned decay schedule, no learned
merge selection beyond the existing content addressing.

## Arms (budget sweep + load-bearing control + native controls)

| arm | budget B | pop | decay | merge | role |
|---|---|---|---|---|---|
| `tape_B0` | B0 = GDN2-equivalent fixed size | yes | yes | yes | fixed-size control |
| `tape_B0x4` | 4×B0 | yes | yes | yes | budget sweep |
| `tape_B0x16` | 16×B0 | yes | yes | yes | budget sweep |
| `tape_unbounded` | machine cap (see T5) | yes | yes | yes | the unbounded fork |
| `tape_appendonly` | machine cap | **NO** | as `tape_unbounded` | as `tape_unbounded` | pop load-bearing control |
| `gdn2_native` | — (fixed matrix state) | — | — | — | native control |
| `e97_native` | — (fixed vector state) | — | — | — | native control |

- **B0 definition (PROPOSED, T1)**: state-cell equivalence with the X27 small
  GDN2 effective state (key128 × value1104 = 141,312 cells, receipted in
  `results_stage27_small_architectures/receipts/parameter_counts.json`).
  With per-slot cost (keydim + valuedim) fp32 cells, B0 = ceil(141,312 /
  (K+V)); the exact K, V dims are bound at commit with a parameter-count and
  state-cell receipt. Bit-equivalence and parameter-equivalence alternatives
  are flagged (T1), not silently chosen.
- **`tape_appendonly` isolates pop** (single-variable vs `tape_unbounded`):
  pop disabled, decay and merge retained. The registered prediction is FAIL on
  `a^k b^k`; see T4 for the alternative "all-reductions-off" reading.
- **Native controls at matched sizes**: `gdn2_native` = the X27 small-GDN2
  config (563,218 params), `e97_native` = the X27 small-E97 config (570,464
  params) — same backbone band as the tape arms. Native OOD failure is the
  RECEIPTED X25/X27 empirical regularity at tested configurations; the
  theorems bound only the asymptotic regime (exact separation at all depths),
  and native finite-depth success is NOT theorem-excluded.
  Size-matching is comparability bookkeeping, not a claim that size matters.
  Theorem citations (asymptotic scope):
  `StreamingExpressivity.balanced_needs_unbounded_state`,
  `MemoryBoundary.context_free_discrimination_impossible`,
  `TapeMemory.tapeModel_no_finite_abstraction`.

## Training curriculum (staging receipts gate progression)

Each stage transition requires its receipt; a stage failure halts that arm
with the stage recorded (no automatic retry; a replacement requires a NEW
registration — X16E/X27 discipline).

1. **Stage 1 — pure associative recall** (reviewed `tasks/mqar_recall.py`
   generator family): delta-rule error-correcting write is the only needed op;
   ignites easily. Receipt gate: >=99% validation balanced accuracy.
2. **Stage 2 — append-necessity tasks** (fresh-key recall /
   `tasks/selective_copy.py` family where no good match exists): the append
   gate is load-bearing. Receipt gate: >=99% validation balanced accuracy AND
   a receipted slot-occupancy measurement showing the store growing with item
   count (the append gate actually firing).
3. **Stage 3 — the ladder** (`a^k b^k`, below): tests pop and budget-
   dependent merge; decay remains inert within the ladder (T2 resolution).
   No extra stage gate; standard adjudication.

Native controls run stage-3 data with the identical recipe; stages 1–2 are
tape-head ignition mechanics that have no native analogue. This asymmetry is a
declared curriculum confound (T6), not a hidden one.

## The instrument: the `a^k b^k` ladder

The asymptotic certificate (Lean, sorry-free): X21T
`balanced_needs_unbounded_state` (via `context_free_discrimination_impossible`)
bounds the total state channel — exact separation at ALL depths requires no
finite total-state abstraction, any dimension, any activation, any budget;
X28T exhibits the constructive object on the other side
(`tapeModel_decides_balanced`, `TapeMemory.tapeModel_no_finite_abstraction`)
with unbounded slot COUNT and bounded per-slot precision. **Finite-depth
scope (review finding, bound):** the formal bound is on DISTINCT WHOLE-
CHANNEL STATE VALUES, not slots or fp32 cells. KEY/VALUE-PAYLOAD-ONLY
illustration (excluding liveness/age/usage metadata and any other
recurrent state): a B-slot store of fp32 (k,v) cells has at most
2^(32·B·(K+V)) distinct payload bit patterns — an explicit upper-bound
illustration that supplies NO certified finite-depth failure location.
**Exact separation at
all depths requires no finite total-state abstraction; X28E tests
finite-depth behavior, and neither its slot-count breakpoint nor finite OOD
success is theorem-certified.** No finite X28E run proves or disproves the
asymptotic statements; the ladder measures where the empirical breakpoint
sits and the finite-depth evidence (occupancy receipts, routing traces)
that supports the growth-mechanism interpretation.

- **Classes balanced from day one** (X26c prior-drift lesson baked in): every
  depth bin is 50/50 accept/reject with fixed marginals; every reported number
  includes balanced accuracy, per-class recalls, confusion matrices, class
  counts and the majority baseline per bin. Reject class = the three X28T
  malformed shapes: leading `b` (`tapeModel_rejects_leading_b`), count
  mismatch `a^j b^k`, j≠k (`tapeModel_rejects_count_mismatch`), trailing `a`
  (`tapeModel_rejects_trailing_a`). Final-only binary decision at end-of-word,
  BCE on a two-logit margin (X23E house supervision); no per-step
  supervision, no teacher forcing.
- **Graded depth bins**: train k ∈ [1..128] (in-range); OOD bins at 2×
  (129–256), 4× (257–512), 9× (1025–1152) of the training-top k=128. Learning
  = >=99% ID and >=99% in EVERY OOD bin (X25/X27 graph standard); ID failure
  and OOD degradation are reported separately with exact counts/losses.
- **Phase-boundary hypothesis (primary deliverable: the
  generalization-vs-budget phase diagram; MECHANICS-predicted, explicitly
  NOT theorem-certified)**: bounded tape arms hold while k <= B and
  collapse past it — an ENGINEERING hypothesis about the head's own budget
  mechanics (merge-forced degradation when append pressure hits the cap),
  not a theorem consequence; no certified finite-depth failure location
  exists (see the finite-depth scope above). The
  three bounded tape arms are each evaluated on a fine per-k sweep in a
  window around their own B (registered granularity: every 16 slots over
  [B−128, B+128]; native arms have no B and no sweep) so the measured cliff
  is located, not just bracketed by coarse bins. T3 records that the theorem
  excludes exact separation at all depths under a finite total-state
  abstraction; it supplies no certified finite-depth failure location for
  X28E and guarantees neither failure at B nor success within B. With
  B0 <= 1024 (T1
  resolution) the 9x bin (k up to 1152) lies entirely beyond the B0 arm's
  slot budget and predicted breakpoint — a clean witness.
- **Unbounded arms are predicted to hold at ALL tested depths within the
  machine memory budget** — the central hypothesis, tested as depth
  generalization (T7 records that the escape-the-regime claim here is about
  depth, not task novelty).

## Pre-registered outcome cells (no post-hoc thresholds)

- **TAPE-LEARNS-AND-GENERALIZES**: `tape_unbounded` reaches >=99% ID and
  >=99% in ALL OOD bins, with receipted slot occupancy growing with k past the
  training regime. Implication: states grow to the size the task demands
  outside the training regime, at tested depths; the unbounded fork is the
  viable production direction. Not an asymptotic proof.
- **TAPE-LEARNS-IN-REGIME-ONLY**: unbounded arm learns ID but drops >=5pp
  ID-to-far-OOD (X24 criterion). Implication: growth capacity was not the
  binding constraint; learning/generalization of the ops is. The mechanism
  claim narrows; no capacity conclusion.
- **NOTHING-LEARNS**: no tape arm learns ID. Implication: task/encoding or
  optimization becomes suspect, NOT an architecture impossibility — the
  native-tokenization control and generator audit fire; that control needs a
  NEW registration.
- **BOUNDARY-MATCHES-THEORY**: bounded arms hold for k <= B and collapse
  past it, with the measured cliff inside the registered window around B.
  Implication: the head's budget mechanics are the binding constraint at
  tested depths — consistent with the MECHANICS hypothesis (explicitly
  NOT theorem-certified; no certified finite-depth failure location). The
  phase diagram is the empirical map of the head's
  own resource limit.
- **BOUNDARY-BETTER-THAN-THEORY**: a bounded arm holds PAST its predicted
  slot-count breakpoint via merge compression (at budget the head is
  GDN2-like, and that softness rescues past the naive breakpoint).
  Implication: POSITIVE result for the
  fixed-big-compressed production stance — bounded plus merge policy beats
  naive bounded; compression buys measured depth beyond the raw slot count.
  Attribution discipline: the routing-trace receipts (merge firing at the
  cliff) SUPPORT merge-causation; the single capped arm does not
  INDEPENDENTLY establish it. Still bounded: the asymptotic theorem stands;
  the rescue is finite-depth and is not a theorem conflict.
- **APPEND-ONLY-FAILS (control)**: `tape_appendonly` fails `a^k b^k` while
  `tape_unbounded` learns it. Implication: pop is load-bearing. If
  append-only ALSO learns, pop is NOT load-bearing at tested depths and the
  mechanism claim is revised, not salvaged.
- **NATIVES-FAIL-OOD (control)**: GDN2 and E97 fail OOD at all tested sizes.
  Implication: consistent with the X25/X27 receipts (native OOD failure is
  an empirical regularity at tested configurations, NOT a theorem
  exclusion at finite depth). If a native GENERALIZES OOD, an instrument
  audit fires — looking empirically for bounded shortcuts — before any
  architecture conclusion; finite-depth native success is not
  theorem-excluded and must not be described as leakage-by-theorem.

Mixed/intermediate outcomes are reported with exact counts/losses per arm and
bin, never forced into a cell. One seed is exploratory. Software/nonfinite
errors are not scientific failures; missing bins, OOM/disk/wall and external
interruptions are right-censored.

## Engineering gates BEFORE any training

1. **Fused-kernel equivalence vs eager fp32 oracle** (X24 gate standard):
   short-sequence fused output/final-state max-relative error < 0.03;
   input/parameter gradient max-relative error < **0.08**; fused eval error
   < 0.03. `receipts/kernel_match.json` binds measured values, seed, zero
   updates. An eager fp32 full-layer oracle exists only for the gate, never
   on a training path.
2. **BPTT through the growing store**: sparse checkpoints + reverse-replay
   per the `ndm/triton/mlp_mem_fused.py` pattern (`_mlp_mem_fwd_kernel` saves
   state every CKPT_INTERVAL steps; `_mlp_mem_bwd_kernel` replays each segment
   forward then walks it in reverse threading the state gradient in
   registers). Pop/decay/merge make the recurrence NON-associative in slot
   space — no chunked/parallel scan; the forward is a sequential per-token
   scan. Checkpoint payloads include slot liveness/age/usage metadata, not
   just keys/values, so replay reproduces allocation decisions exactly.
   CKPT_INTERVAL bound at commit.
3. **Precision (X21E/X24 legacy receipt)**: fp32 live slot state, bf16
   backward replay checkpoints/scratch, fp32 parameters. Not bitwise parity;
   a declared numerical tolerance, not a new precision variable.
4. **Numerics smoke at 90,904 tokens** (house max-length standard): fresh
   zero-update max-length forward/backward + fused evaluation per arm, finite
   gradients, receipted peak allocation. Unbounded arms add a cap-stress
   smoke at the machine slot cap (T5).
5. **Nonfinite-death guard (X27 softplus-small receipt)**: an arm dying
   nonfinite (logits or gradients) is **CENSORED infrastructure, never a
   training verdict**. Fail-closed: the arm is not adjudicated failed-to-learn;
   the error receipt is retained; any replacement arm requires a NEW
   registration.

## Resource plan

- **Cards**: physical card 7 is EXCLUDED — the X25 big-E97-softplus run is
  live there; it is left alone. Cards 0–6 are free at draft time
  (`scripts/gpu_lease.sh list-free` receipt above). Seven training arms map
  one-per-free-card as PREFERENCES only; the X27q amendment policy applies:
  registered card if free at launch, else next-free ascending physical index,
  actual card receipted per-arm launch receipt; atomic idle/unleased shell
  leases only, no preemption, no WorksGood.
- **Seeds**: n=1 (seed0) per cell, right-censoring at the 60,000-update cap
  retained; no claim that the cap is a training ceiling. No extra
  seeds/tasks/recipe retries authorized.
- **Standard chain**: registration/code/gates/generator/dependency identities
  SHA-bound and committed+pushed before launch; trainer verifies worktree AND
  HEAD; disk guard >=400GiB free, <=40GiB per rung, wall <=7 days;
  validation-only checkpoint selection; independent 60s persistent watcher,
  identity-verified owned process groups, TERM then KILL after 30s;
  hash-checked result/terminal/marker receipts; watcher wiring receipted for
  supervisor native arming.

Local lambda01 CUDA research, not Frontier and not resilient DiLoCo:
R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 do not apply; no Slurm
Partition/QOS binding or resilient conformance claimed. Independent acceptance
review required before commit; nothing here is a launch authorization.

## Design tensions (FLAGGED, not resolved)

- **T1 — "GDN2-equivalent" is ambiguous** (state cells vs bits vs parameters).
  Proposed: cell-count equivalence (141,312 cells). Alternatives change B0 and
  hence where the B0 cliff is predicted. RESOLVED by operator delegation
  (see Tension resolutions): state-cell equivalence; B0 <= 1024 with the
  freeze receipt binding dims/multiplicity/rounding (see T1/B0 arithmetic).
- **T2 — usage-decay can kill the UNBOUNDED arm at depth by policy, not
  capacity.** Long `a`-runs can decay early slots before their `b`s arrive;
  the central-hypothesis arm would then fail for cache-policy reasons. Decay
  constants are pre-registered and fixed, but their values are
  outcome-relevant: a decay-driven deep-k failure must be reported as
  cache-policy-binding, distinct from TAPE-LEARNS-IN-REGIME-ONLY.
- **T3 — the theorem excludes exact separation at all depths under a
  finite total-state abstraction; it supplies no certified finite-depth
  failure location for X28E and guarantees neither failure at B nor
  success within B.**
  "Holds while k <= B" is an engineering prediction; learnability within
  budget is the open empirical question the sweep measures.
- **T4 — the append-only control's op set defines what it proves.**
  Registered: pop disabled, decay+merge retained (single-variable vs
  `tape_unbounded`). The stricter "all reduction off" reading is a different
  control; if the operator wants it, it is an eighth arm, not a reinterpretation.
- **T5 — "unbounded" means a machine cap.** Proposed cap: 2^17 = 131,072
  slots per (batch-element, head), receipted at launch with the cap-stress
  smoke. "Holds at ALL tested depths" is conditional on this cap; hitting the
  cap is right-censored, not a scientific failure.
- **T6 — curriculum asymmetry confounds tape-vs-native comparison.** Stages
  1–2 are tape-ignition scaffolding natives cannot consume; cross-arm
  comparisons at stage 3 carry this declared confound (X27 concession-19
  pattern), qualitative only.
- **T7 — "outside the training regime" is tested as DEPTH here.** The ladder
  tests depth generalization, not task novelty; the central hypothesis is
  operationalized on one axis and the registration claims only that axis.
- **T8 — SUPERSEDED by the T1 resolution and the T1/B0 arithmetic section**
  (the original text assumed B0 ≈ 1104 cells-as-slots and a straddling
  9× bin; the resolution binds B0 <= 1024 SLOTS so the 9× bin lies entirely
  beyond the B0 arm's predicted breakpoint; the fine sweep locates each
  bounded arm's measured cliff regardless). Retained for provenance.
- **T9 — delta-rule default vs pure append on the a-phase.** The delta-rule
  best-match overwrite is the ignition receipt of stage 1, but `a^k b^k`
  semantically wants every `a` to append (identical content; no error to
  correct). The match threshold governs which op fires; it is a fixed
  architectural constant whose value is outcome-relevant for the ladder. A
  stage-3 failure traceable to overwrite-instead-of-append is a mechanism
  finding, reported as such.

## Tension resolutions (operator-delegated, bound at commit)

The operator delegated the flagged design decisions to the supervisor with
the directive, verbatim: "Okay, this is all a bit wild, but I'm happy for
you to do these experiments if you think that they're useful."
Resolutions as bound:

- **T1 — B0 = STATE-CELL EQUIVALENCE**: 141,312 fp32 state cells (matched to
  the X27 small-GDN2 receipt). Cells are a natural COMPARABILITY axis — the
  same state-cell budget as the native control — but NOT the theorem's axis
  (the theorems bound DISTINCT WHOLE-CHANNEL STATE VALUES; see the
  finite-depth scope in the instrument section). Bit- and parameter-
  equivalence are recorded as rejected alternatives with this reasoning.
  Slot key/value dims are bound at the engineering freeze such that
  **B0 <= 1024 slots**, keeping the 9x OOD bin (k <= 1152) entirely beyond
  the B0 arm's slot budget and predicted breakpoint (T8 corollary); the fine
  per-k window sweep locates each bounded arm's measured cliff regardless.
- **T2 — decay inert within the ladder**: registered fade horizon >= 10x the
  longest tested word, calculation receipted at commit (spec bullet amended).
- **T3 — stands as caveat**: the theorem excludes exact separation at all
  depths under a finite total-state abstraction; it supplies no certified
  finite-depth failure location for X28E and guarantees neither failure at
  B nor success within B; "holds while k <= B" is the engineering
  prediction the sweep measures.
- **T4 — append-only control is POP-ONLY-OFF** (decay + merge retained,
  single-variable vs `tape_unbounded`). The stricter all-reductions-off arm
  is NOT added (eighth-arm recommendation declined by supervisor delegation;
  the load-bearing claim does not need it and the card budget is fully
  allocated at seven arms).
- **T5 — unbounded = 2^17 = 131,072 slots per (batch-element, head)**, cap
  receipted at launch with the cap-stress smoke; cap-hit right-censored.
- **T9 — APPEND WHEN THERE IS NOTHING TO CORRECT**: growth fires on zero
  write-error (identical content accumulates; conflicting content reconciles
  via the delta rule). Spec bullet amended. A stage-3 failure traceable to
  overwrite-instead-of-append remains a reportable mechanism finding.
- **T6, T7 — stand as declared caveats** (curriculum asymmetry declared,
  escape-the-regime claimed on the depth axis only).

## The routing rule (complete executable specification — T9 closed)

**Per-token operation dispatch, in order**: per token, per (layer, head),
the head first performs a **READ** — which may carry a learned pop strength,
and a POP of the best-matched live slot is a READ-PATH action recorded in
the trace (pop fires when the pop strength exceeds `pop_threshold`, bound
at freeze) — and then a **WRITE-ENABLE decision** (rule and constants
frozen at M2): the token may be routed **W-OFF** (no write at all; traced
as NO-WRITE), or **W-ON**, in which case the write routes to exactly one
of five write routings below. A token may therefore pop without writing,
write without popping, both, or neither. The trace records POP and WRITE
target slot indices SEPARATELY when both occur (their targets need not
coincide). All thresholds (`match_threshold`, `eps_residual`,
`pop_threshold`, the write-enable rule) are FIXED ARCHITECTURAL CONSTANTS
bound in the engineering
freeze with their values receipted; the learned gate strengths modulate
write/pop magnitudes but CANNOT override routing — routing is
architectural, always:

- **R-EMPTY**: no live slots exist → **APPEND** a new slot (allocation
  conditional on remaining capacity — see R-CAP).
- **R-NOMATCH**: best-match score < `match_threshold` → **APPEND** (this is
  the stage-2 fresh-key case: novel bindings append; the earlier
  stage-2/R-CONFLICT conflict is resolved here — a fresh key has no good
  match, so it appends, never corrects).
- **R-ZERO-RESIDUAL**: best-match score >= `match_threshold` AND squared
  value residual < `eps_residual` → **APPEND** (the nothing-to-correct case;
  architecturally forced — identical content accumulates).
- **R-CONFLICT**: best-match score >= `match_threshold` AND residual >=
  `eps_residual` → **DELTA-RULE CORRECTION** on the matched slot
  (error-correcting write; the stage-1 ignition op).
- **R-CAP — FINAL OVERRIDE**: any APPEND routing (R-EMPTY/R-NOMATCH/
  R-ZERO-RESIDUAL) requested while the budget B is full resolves to
  **MERGE** (delta-rule merge onto the best match; no slot is ever allocated
  past B). R-CAP takes precedence over the append request; routing is
  evaluated in the order listed.

**Ladder honesty (re-review correction, bound):** Every `a` requesting
append is a registered mechanism prediction, not an architectural
guarantee. Conditional on W-ON, R-ZERO-RESIDUAL requests append; allocation
occurs only when capacity remains, otherwise R-CAP merges. Identical
learned write content alone does not guarantee append allocation — nothing
architecturally forces the learned key/value projections to produce
identical content at different positions, and W-OFF suppresses the write
entirely. The prediction is receipted by the routing traces and the
peak-occupancy measurements.

**Zero-update trace obligations (engineering-freeze gate):** traced runs
receipting every routing case firing correctly — empty-store bootstrap
(R-EMPTY), repeated identical writes (R-ZERO-RESIDUAL), novel bindings
(R-NOMATCH), conflicting corrections (R-CONFLICT), cap-pressure merge
(R-CAP), and pop-on-read — before any optimizer update.

## Frozen recipe and data contract (scientific choices — bound now)

**Inherited recipe (X25/X27 standard, exceptions listed):** AdamW, betas
(0.9, 0.999), eps 1e-8, weight-decay 1e-4, lr 1e-4 constant, batch 64,
gradient clip 1.0. Exceptions: none at draft; the ENTIRE scientific
recipe/data contract is FROZEN AT M2 (the engineering freeze) — after M2 no
recipe or data change may enter by amendment at any stage; any deviation
requires a NEW separately-reviewed registration that cannot inherit this
one's primary-registration claim unchanged.

**Stage budgets (right-censored, no retries, no post-hoc extension):**
stage 1 = 15,000 updates; stage 2 = 15,000; stage 3 = 30,000; total cap
60,000 (the house cap). A stage may complete early on its receipt gate;
unused budget does not carry forward.

**Evaluation:** validation every 500 updates on the arm's stage validation
pool; checkpoint selection validation-only, tie-break latest-best; the
selected checkpoint is the ONLY one evaluated on test/OOD bins (immutable
pools, below).

**Data (immutable, seed-fixed at freeze):** stage-1/stage-2 generator files
(`tasks/mqar_recall.py`, `tasks/selective_copy.py`) with registered
vocabularies, sequence-length bands, per-class counts and split seeds
receipted at freeze; stage-3 ladder: train pool k ∈ [1..128] balanced
50/50 accept/reject; per-depth-bin evaluation pools of 512 balanced words
(bin = ID/2x/4x/9x, fixed marginals, fixed seed), never trained on; fine
per-k sweep points get 256 balanced words each. Reject-class composition:
the three X28T malformed shapes equally weighted, sampled within the bin's
depth band, per-class counts receipted. Tokenization: reviewed scalar-token
convention (the `merrill26_separators.py` standard), bound at freeze.
Stage-1/2 metrics: mean per-class recall (multiclass balanced accuracy);
stage-3: binary balanced accuracy + per-class recalls + confusion matrices
+ majority baseline per bin (X26c discipline).

**Milestones (two, explicit):** M1 = registration acceptance (this document,
reviewed and committed) authorizes ENGINEERING ONLY — kernels, oracle,
traces, smokes, gate receipts — with a reviewed, pushed ENGINEERING FREEZE
(code/config/gates/generators, SHA-bound) required as M2 before ANY
curriculum optimizer update. No stage-1 update exists before M2.

## Adjudication rules (registered, no post-hoc thresholds)

- **Per-k metrics**: at each fine-sweep k and each bin, the metric is binary
  balanced accuracy on that pool (512 words/bin, 256/sweep-point).
- **HOLD at k**: balanced accuracy >= 0.99. **COLLAPSE at k**: >= 5pp drop
  from the arm's ID accuracy (X24 criterion). Bins/sweep points are
  adjudicated independently; intermediate values are reported with exact
  counts, never forced.
- **Cliff location (bounded arms)**: an ORDERED bracket is required — the
  largest sweep-point k with HOLD must precede the smallest sweep-point k
  with COLLAPSE (both inside the registered window [B−128, B+128]); the
  cliff is the midpoint between those ordered endpoints, reported with both
  endpoints' counts. The cliff is "inside the window" iff both endpoints
  exist and are ordered. NON-MONOTONIC sweeps (COLLAPSE before HOLD, or
  multiple HOLD/COLLAPSE transitions) are NOT assigned an ordinary cliff:
  they are classified NON-MONOTONIC, reported with the full per-k curve
  and exact counts, and excluded from the phase-diagram cliff fit.
- **Occupancy evidence (growth receipts)**: PEAK live-slot count during the
  word, receipted per k; growth-in-k at OOD = peak occupancy tracking k
  within the registered tolerance (peak ∈ [0.9k, 1.1k]) at OOD depths.
  End-of-word occupancy ~0 on accepts is the pop receipt, reported
  alongside.
- **Censored/partial arms**: excluded from joint cells, reported
  separately with the censor receipt (nonfinite = CENSORED infra, X27
  standard; cap-hit on unbounded arms = right-censored).

## Replay-decision consistency gate (P1-6, engineering-freeze obligation)

bf16 replay checkpoints cannot guarantee identical routing decisions
(a live fp32 value 1.0001 rounding to 1.0 in bf16 can flip R-ZERO-RESIDUAL
to R-CONFLICT between forward and replay). Obligation, bound at freeze:
**the forward pass saves a discrete ROUTING TRACE** (per token, per head:
which of R-EMPTY/R-NOMATCH/R-ZERO-RESIDUAL/R-CONFLICT/R-CAP fired and
the matched slot index, or NO-WRITE, plus the POP decision and its target) in the checkpoint payload; **replay follows the
trace** — routing decisions are replay INPUTS, never recomputed from
rounded state. A multi-checkpoint branch-equivalence gate receipts that
replayed routing equals the forward trace at every checkpoint boundary,
covering all routing cases. Liveness/age/usage metadata alone is explicitly
INSUFFICIENT and the earlier promise to that effect is superseded by this
gate.

## T1/B0 arithmetic (P2-7 reconciliation)

B0 = ceil(141,312 / (K+V)) slots with K+V >= 138 required for B0 <= 1024
(e.g. K=64, V=128 → K+V=192 → B0 = 736 slots). Cell equivalence is
APPROXIMATE under ceiling division; the engineering-freeze receipt binds
the actual dims, head multiplicity, the resulting B0, the exact cell total,
the rounding, and the excluded bookkeeping metadata (liveness/age/usage are
NOT counted as state cells). The earlier "B0 ≈ 1104 / 9x straddles" wording
is superseded: with B0 <= 1024 the 9x bin lies beyond B0's slot budget
  and predicted breakpoint
entirely. The fine sweep applies to the three bounded tape arms only.

## T2 calculation obligation (bound at freeze)

The decay-inertness calculation must use the ACTUAL longest registered
input — including malformed words, the 90,904-token smoke, and the
cap-stress smoke — and show the decay/free equations proving a live slot's
fade horizon >= 10x that length (at the 9x bin, a positive word is 2,304
tokens; horizon >= 23,040 tokens). The calculation is a freeze receipt; a
decay-driven deep-k failure is reported as cache-policy-binding, distinct
from TAPE-LEARNS-IN-REGIME-ONLY.

---

## M2-review-driven amendments (APPEND-ONLY — every original section above is preserved verbatim)

**Amendment F5 (M2-review finding 5 — registered effective-batch
accumulation).** batch 64 = effective batch 64 via length-aware microbatch
accumulation; per-arm token budgets are freeze-bound receipts; updates are
identical to a single batch-64 step. Each optimizer step consumes exactly 64
examples; the 64 examples are processed in length-aware microbatches bound
per arm in `configs/<arm>.json` (`microbatch.token_budget` bound on the
padded allocation product n_mb * max_length, plus
`microbatch.max_sequences` for the fixed per-sequence backward-replay-cache
cost of the unbounded arms' 2^17-slot stores), chosen so the predicted peak
allocation fits a 40-GiB rung with margin and receipted in
`receipts/microbatch_allocation_model.json`; the summed loss over the 64
examples is accumulated across microbatches and ONE optimizer step fires per
64 examples — mathematically the registered single batch-64 update (pure
gradient-equivalence test in `eng/tests/test_trainer_wiring.py`). A single
example exceeding the token budget forms its own microbatch (over-plan
single, receipted); a projected above-rung single example is CENSORED
infrastructure (fail-closed, never a scientific verdict). Registration gate
4 is discharged PER ARM by the 90,904-token zero-update
forward/backward/evaluation smokes at the bound budgets
(`eng/fused/receipts/per_arm_smokes/<arm>.json`); the cap-stress smoke stays
for the two cap arms (`eng/fused/receipts/cap_stress_smoke.json`).

**Amendment (a) (M2-review adjudication (a) — precision-surface
clarification, required before freeze).** The precision surface is bf16
sparse checkpoints as the declared rounding surface with an fp32 per-step
replay cache; trace-following replay makes the cache's precision
non-decisional; the receipted bf16-path gradient parity (2.16e-3/2.12e-3
<< 0.08) supports accepting the implementation as-is. This clarifies gate 3's
"bf16 backward replay checkpoints/scratch" wording: the sparse checkpoints —
the surface where rounding threatens decision consistency — are bf16 as
registered; the per-step replay cache inside `tape_bwd_kernel` is fp32; the
G10 trace-following replay consumes discrete routing/free decisions as
inputs, so the cache's precision cannot flip any architectural decision.
No bf16-cache rewrite is required by the current evidence.

## M2-review-2-driven amendment (APPEND-ONLY — originals above preserved)

**Gate 4 (numerics smoke) restated per the registered accumulation mechanism
(M2-review-2 finding 4; the F5 effective-batch amendment made the split form
the registered mechanism, and this amendment makes the gate's wording match
what the mechanism implies — explicitly, not by relabeling receipts):**

- The per-arm gate-4 obligation is: a zero-update forward/backward/eval
  smoke covering the registered 90,904-token count THROUGH THE ARM'S FROZEN
  MICROBATCH BUDGET (the effective-batch accumulation mechanism registered
  above), finite throughout, within the 40-GiB rung, with per-arm receipts.
- ARMS WHOSE FROZEN BUDGET ADMITS A LITERAL SINGLE 90,904-TOKEN TRAJECTORY
  ADDITIONALLY RUN IT and receipt it as the literal max-length form:
  `tape_B0`, `gdn2_native`, `e97_native` receipt literal single-trajectory
  90,904-token smokes (n_microbatches = 1 in
  `eng/fused/receipts/per_arm_smokes/*.json`).
- ARMS THAT SPLIT (tape_B0x4, tape_B0x16, tape_unbounded,
  tape_appendonly) receipt the aggregate-token form at their frozen budgets.
  For the two cap arms a literal trajectory is INFEASIBLE within the rung:
  the allocation-model receipts show a single 2,306-token over-plan at
  36.1 GiB for a 131,072-slot store, scaling far past the 40-GiB rung at
  90,904 tokens; the receipts in `receipts/microbatch_allocation_model.json`
  are the infeasibility evidence, and the aggregate form is the ONLY
  feasible registered discharge for those arms.
- No receipt is labeled as a literal max-length trajectory unless its
  n_microbatches is 1.

**Amendment F5 epoch-boundary policy (M2-review-2 finding 5).** The
exactly-64 rule binds the sampler: the trailing partial batch (48 examples
at the 70,000-example pools) is DROPPED at each epoch boundary; every epoch
re-permutes the full pool with the registered seed sequence, so the dropped
remainder differs per epoch and no example is starved across epochs. The
sampler is deterministic and unit-tested at 70,000 examples.

---

## Amendment F12 (post-M2-launch ignition correction; supervisor-delegated
## "make this work"; append-only — the original terms above stand as written)

### Receipted failure being amended

The M2-frozen package launched; the tape arms halted at the stage-1 gate
(tape_B0 and tape_B0x4: `halt_stage_gate_failed`, stage1 metric 0.0697,
validation mean-per-class-recall 0.0155 = the 1/64 random floor for the
64-way MQAR query). The natives were unaffected (gdn2_native completed at
0.9915 balanced accuracy; e97_native in stage 3 at 0.99), proving the
trainer loop, generators, sampler, and optimizer correct and isolating the
defect to the tape head's ignition dynamics.

### Diagnosis (probe trail, all receipts in the session log)

1. Trained-checkpoint routing histogram: 96.8% NO_WRITE, 1.9% appends —
   the tape was a closed library; reads returned noise.
2. The write gate initialized AT its threshold (sigmoid(0) = 0.5 =
   `write_enable_threshold`); early writes are noise; the gradient
   through the correction magnitude closes the gate; a closed gate is an
   ABSORBING state (no corrections fire -> no gradient to reopen).
3. The pair binding (key->value) required a chance match of the value
   token onto its key slot (~1/S) with detached argmax aiming: the
   gradient is majority-noise. The architecture as originally wired is
   depth-1 with stateless projections: MQAR cannot bootstrap.
4. With a two-token windowed MATCH key, binding became deterministic —
   but the read gradient then collapsed the stored-key geometry (key
   appends decayed 3035->2547 while conflicts rose; accuracy decayed).
5. Final recipe: stable 0.817 validation accuracy at 500 updates, stable
   through 3000 (probe 11), gates passing, all numerics gates passing.

### The five registered changes (F12a-F12e)

- **F12a — split stored/match keys with a two-token window**: stored keys
  are `W_k(x_t)` (single-token, unchanged storage semantics); the write
  SEARCHES with `k_match = W_k(x_t) + W_k(x_{t-1})` (shared W_k, zero new
  parameters). Value tokens align with their pair's key slot
  deterministically at init; stored keys never contain windowed
  components, so token repeats cannot create spurious matches. Ladder
  note: consecutive identical `a` tokens produce `2*W_k(a)` match keys
  with zero value residual -> R-ZERO-RESIDUAL appends: the registered
  every-a-appends mechanism prediction now fires architecturally.
- **F12b — separate match scale**: write-match scores carry their own
  scale (1/3 of the read scale, i.e. 1/(3*sqrt(K))); the read softmax
  keeps 1/sqrt(K). The registered `match_threshold` semantics become
  reachable and meaningful (own-slot ~0.9, spurious ~4-sigma tail).
- **F12c — architectural write gate + full-overwrite correction**: the
  write gate is DETACHED (sigmoid > threshold, masked off for the pad/
  gap token id 0 per the generator's own "never a supervised target"
  contract, bias initialized +2 = open on content) — zero learned
  management, per the registered spine. The R-CONFLICT correction is the
  FULL overwrite the original registration text already specified
  ("error-correcting overwrite of the matched slot's value"); the gate's
  magnitude no longer appears in the attached path, eliminating the
  absorbing collapse channel.
- **F12d — frozen key geometry**: stored and match keys are DETACHED from
  the read gradient. The read path cannot collapse the addressing space;
  queries (W_q) and values (W_v) adapt to a frozen random geometry.
  W_k's gradient is expected ABSENT (the F10 W_pop precedent).
- **F12e — retrieval warm start**: W_q initialized equal to W_k, so a
  query finds its own token's slot from step 0.

### Unchanged

All five routing cases and their order, thresholds, budget arithmetic
(B0 = 736 slots), pop/decay/merge reductions, zero-learned-management
spine, stage structure and budgets, gates, ladder, arms, outcome cells,
natives (untouched construction), eval protocol, and every M2
verification chain. The stage-1 mechanism ceiling (~0.85, from
value-collision ties through single-token value appends) is a reportable
mechanism finding, not a defect.

### Disposition of pre-F12 runs

The pre-F12 tape-arm results (2x halt receipts + 3 superseded crawls) are
archived under `*_seed0_preF12` as the amendment's evidence; the natives
re-run under the F12 freeze for one registration lineage. No pre-F12
optimizer update counts toward any F12 arm.
