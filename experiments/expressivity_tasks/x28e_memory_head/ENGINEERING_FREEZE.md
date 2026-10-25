# X28E ENGINEERING FREEZE (M2 package) — the budgeted-growth tape head

**Status: M2 freeze package assembled; UNREVIEWED, UNCOMMITTED, UNPUSHED at
assembly time. Zero X28E optimizer updates exist.** Binding ancestry: the
registration (REGISTRATION.md, accepted at M1 as commit `574bf399`) makes M2
a reviewed, pushed freeze of code/config/gates/generators, SHA-bound in
`freeze_manifest.json` (this directory), BEFORE any curriculum optimizer
update. Nothing in this package is a launch authorization; nothing trains
before the M2 review, commit and push. The trainer mechanically enforces
this: no optimizer update is reachable unless `--frozen-freeze-hash`
matches the manifest hash, with worktree AND HEAD verification (standard
chain).

Contents: (1) the freeze table; (2) the G1–G14 resolutions restated as
BOUND; (3) the phase-1 liveness-bug correction; (4) the precision surface as
implemented, with its FLAG; (5) the T2 recalculation; (6) the cap receipt;
(7) the frozen recipe/data contract and arm table; (8) the package
inventory and manifest rule; (9) residual M2-review audit items — flagged,
never silent.

---

## 1. The freeze table — every bound value with its receipt reference

| quantity | BOUND value | receipt / reasoning reference |
|---|---|---|
| d_model | **256** | G7 freeze dims; `eng/fused/receipts/kernel_match.json:dims`, `numerics_smoke_max_length.json:dims` |
| NH (head multiplicity) | **4** | G7; same receipts |
| K (key dim) | **64** | G7; same receipts |
| V (value dim) | **128** | G7; same receipts |
| K+V | **192** (>= 138 required for B0 <= 1024, T1) | T1/B0 arithmetic (REGISTRATION); `kernel_match.json:dims` |
| **B0** | **736 slots, EXACT** — `B0 = ceil(141,312 / (K+V)) = ceil(141312/192) = 736`; 192 x 736 = 141,312, no ceiling remainder | X27 small-GDN2 state-cell receipt `results_stage27_small_architectures/receipts/parameter_counts.json` (key128 x value1104 = 141,312); `kernel_match.json:dims.B0` |
| match_threshold | **0.5** | G14; `kernel_match.json:constants_G14` |
| eps_residual | **1e-6** | G14; same |
| pop_threshold | **0.5** | G14; same |
| write_enable_threshold | **0.5** | G14; same (supersedes the phase-1 0.25 default) |
| CKPT_INTERVAL | **16** | G14; `kernel_match.json`, `numerics_smoke_max_length.json:dims`, `cap_stress_smoke.json:dims` |
| decay_rate | **0.999995** | G14 + T2 (section 5); `numerics_smoke_max_length.json:t2_fade_horizon` |
| free_threshold | **1e-3** | G14 + T2; same receipts |
| usage_decay | **0.999** (receipt-only bookkeeping, G13; no routing role) | `tape_head.py:TapeHeadConfig`, gates/smokes configs |
| MAX_RESIDENT_SLOTS | **64** (register-resident up to 64 slots; spilled beyond) | `tape_kernels.py:DEFAULT_MAX_RESIDENT_SLOTS`; both modes receipted in `kernel_match.json` (resident S=48, spilled S=736) |
| TILE_S (spilled slot tile) | **64** | `tape_kernels.py:DEFAULT_TILE_S`; spilled-mode receipts above |
| checkpoint_dtype (replay) | **bfloat16** (the declared rounding surface; see section 4) | `kernel_match.json` bf16 receipts; `numerics_smoke_max_length.json` |
| unbounded cap (T5) | **2^17 = 131,072 slots per (batch element, head)** | `cap_stress_smoke.json` (section 6) |
| optimizer | AdamW, betas **(0.9, 0.999)**, eps **1e-8**, weight-decay **1e-4**, lr **1e-4 CONSTANT** | frozen recipe (X25/X27 inherited standard, exceptions: none) |
| batch size | **64** | frozen recipe |
| gradient clip | **1.0** | frozen recipe |
| stage budgets | stage1 **15,000** / stage2 **15,000** / stage3 **30,000** updates; total cap **60,000** | frozen recipe (right-censored, no retries, no carry-forward) |
| eval cadence | **500 updates** | frozen recipe |
| seed | **0** (n=1 per cell) | resource plan |
| stage gates | stage1: **>= 99%** validation mean per-class recall; stage2: **>= 99%** AND the slot-occupancy growth receipt (strictly increasing mean peak live slots across K in {2,4,8,16}); stage3: none (standard adjudication) | REGISTRATION curriculum |
| fine sweep (bounded tape arms only) | every **16** slots over **[B-128, B+128]**, 17 points per arm, **256** balanced words/point | registration instrument section; `configs/*.json:fine_sweep` |
| OOD bins | ID [1..128], 2x [129..256], 4x [257..512], 9x [1025..1152]; **512** balanced words/bin, immutable, never trained on | REGISTRATION; `generators/stage3_ladder.py`, hashes in `generators/receipts/determinism_check.json` |
| cards | **0–6 only; card 7 EXCLUDED** (live X25 run) | resource plan; `scripts/gpu_lease.sh list-free` receipt at draft time; per-arm preferences in `configs/*.json` |
| checkpoint selection | validation-only; tie-break **latest-best** (`>=`) | frozen data contract |

## 2. The G1–G14 resolutions — restated as BOUND, with reasoning

The supervisor gap resolutions from phases 1–2 are binding at this freeze
(restated from `eng/fused/tape_kernels.py`, `eng/fused/run_gates.py`,
`eng/fused/run_smokes.py` and the gate receipts):

* **G1 — POP = strength-decrement** (sigma <- sigma * (1 - p), death at
  `free_threshold`), not binary-kill. Reasoning: keeps the pop magnitude
  learnable per the registration ("the gradient touches ONLY keys, values,
  queries, and the write/pop/append strengths"); the binary-kill reading
  severs that differentiable path; death reuses the registered
  strength-death rule. Receipts: `test_pop_on_read_full_kill_and_reuse`,
  `test_pop_target_separate_from_write_target_and_partial_pop`,
  `test_fused_pop_kills_then_reuses_freed_slot`,
  `replay_consistency.json` (FREE-BY-POP coverage).
* **G2 — soft-strength reads**: live slots are weighted by their
  (differentiable) strength in the read mixture; dead slots are excluded
  exactly via the live mask. Reasoning: this is what gives the pop/decay
  magnitudes a differentiable effect on reads while hard liveness governs
  every routing decision.
* **G3 — write-enable rule**: W-ON iff the write-strength gate exceeds
  `write_enable_threshold` (0.5, bound); W-OFF traced NO-WRITE; the same
  gate value doubles as the delta-rule write magnitude beta. Reasoning: one
  learned scalar serves the registered threshold comparison and the
  registered write strength; the comparison itself is architectural
  (detached), preserving zero learned management.
* **G4 — append gate**: the appended value is scaled by the learned append
  gate whose input includes the DETACHED write-error magnitude against the
  best-matching live slot. Reasoning: registration verbatim ("the gate's
  input includes the write-error magnitude"); the detach keeps routing
  non-differentiable.
* **G5 — budget-full = no FREE slot**: dead slots are reusable free
  capacity; no slot is ever allocated past B; append pressure at a full
  budget resolves to R-CAP merge. Receipts:
  `test_r_cap_merge`, `test_fused_all_cases_and_cap` (alloc < B asserted),
  `cap_stress_smoke.json:alloc_below_cap`.
* **G6 — match scoring**: scaled dot product (q.k)/sqrt(K) for reads, pops
  and write routing, uniformly.
* **G7 — freeze dims**: d_model=256, NH=4, K=64, V=128, hence B0=736 exact.
  Reasoning: satisfies the T1 resolution's B0 <= 1024 slots with K+V >= 138,
  putting the entire 9x OOD bin (k up to 1152) beyond the B0 arm's slot
  budget and predicted breakpoint (T8 corollary); these are the dims at
  which every gate and smoke was receipted. Receipt: `kernel_match.json`.
* **G8 — supervision surface**: end-of-word BCE on a two-logit margin
  reading the FINAL READ output (the final token's layer output is
  W_o(final read) + residual); read-before-write makes the final token's
  own write gradient-dead. Receipts:
  `test_final_token_write_is_gradient_dead_without_later_read`,
  `test_final_token_write_is_gradient_dead_fused`,
  `numerics_smoke_max_length.json:g8_final_token_write_grad_dead`.
* **G9 — per-token op order**: READ -> POP -> WRITE routing; a pop can
  free the last live slot before write routing, so R-EMPTY can fire
  post-pop and the freed slot is reused by the next append. Receipts:
  `test_fused_pop_kills_then_reuses_freed_slot`,
  `test_fused_all_cases_and_cap` (t5), `test_fused_replay_all_cases_at_boundaries`.
* **G10 — free-by-decay traced**: strength-death decisions join the
  routing trace as int8 codes (FREED_POP=1 at the post-pop check,
  FREED_DECAY=2 at the decay check); replay consumes them as inputs and
  never recomputes a death decision from (rounded) state. Receipts:
  `replay_consistency.json` (the engineered would-flip construction:
  fp32 strength 0.5009 survives the forward, bf16 rounds it to 0.5 <=
  free_threshold, recomputation WOULD free it, trace-following keeps
  liveness exact), `test_bf16_replay_trace_consistency_would_flip_free_by_decay`,
  randomized-store parity tests.
* **G11 — T2 recalculation obligation: DISCHARGED at this freeze** (section
  5): the calculation is redone against the actual longest registered
  input of the freeze — the 90,904-token max-length numerics smoke — with
  the equations and the >= 10x proof below.
* **G12 — layer integration**: output projection W_o over the concatenated
  per-head read outputs plus the residual (eager oracle and fused layer are
  state-dict identical). Reasoning: the registration does not bind the
  layer shape; this is the minimal house-standard form, bound at freeze.
  Receipt: `oracle.py`/`fused_layer.py` parity in `kernel_match.json`.
* **G13 — usage counters are receipt-only**: the accumulated read-mass
  bookkeeping has no routing role; freeing is strength-driven (G1/G10).
  `usage_decay = 0.999` is bound as an implemented bookkeeping constant.
* **G14 — the placeholder constants are BOUND** at the values in the
  freeze table (match_threshold=0.5, eps_residual=1e-6, pop_threshold=0.5,
  write_enable_threshold=0.5, CKPT_INTERVAL=16, decay_rate=0.999995,
  free_threshold=1e-3); every gate/smoke receipt was produced at exactly
  these values (`kernel_match.json:constants_G14`).

## 3. Phase-1 liveness-bug correction (recorded, with test receipt)

The phase-1 eager scan contained a broadcast bug in the APPEND liveness
update: `live = live | append_fires.unsqueeze(-1)` broadcast a `[B,NH,1]`
mask over `[B,NH,S]`, marking **every** slot live whenever any append fired.
It was invisible to the phase-1 tests (fresh stores: wrongly-revived slots
carry strength 0 and die again at the end-of-token free check) but corrupts
any store with dead-but-strong slots. The correction revives exactly the
allocated slot (the allocation one-hot is already masked by the append
fire): `live = live | onehot_alloc.to(torch.bool)` — see the
`PHASE-2 PARITY FIX` comment in `eng/tape_head.py`.

**Test receipts that pin the correction**: the randomized-store parity
tests run stores with ~40–50% dead slots and live strengths spanning
[0.05, 1.0] — dead-but-strong slots abound, and exact live-mask/age parity
vs the fused kernels is asserted: `test_routing_parity_random_stores`
(6 instances: seeds 0–2 x resident/spilled),
`test_routing_parity_ragged_tail_spilled`, and the phase-2 gate rerun
`replay_consistency.json:randomized_store_bf16` (130-slot store,
`boundary_live_exact: true` with 13 pop-frees and 6 decay-frees firing).
Both test suites are green at the freeze (34/34; section 8).

## 4. Precision surface — stated exactly as implemented (FLAGGED)

As implemented and receipted:

* **fp32** live slot state, forward kernel arithmetic, all backward VJP
  arithmetic, parameters and inputs;
* **bf16 SPARSE REPLAY CHECKPOINTS** — the declared rounding surface the
  backward replay consumes across segment boundaries (configurable; the
  gates receipt both the bf16 training path and an fp32-exact diagnostic:
  `kernel_match.json:spilled_bf16_ckpt` vs `spilled_fp32_ckpt_diagnostic`);
* **fp32 PER-STEP REPLAY CACHE inside `tape_bwd_kernel`** — the per-program
  HBM scratch (`cache_k/v/s/l/a`) that holds the segment's per-step store
  states during the reverse walk is fp32.

**FLAG — M2-review audit item (against registration gate 3's wording):**
gate 3 reads "bf16 backward replay checkpoints/scratch", grouping the
checkpoints and the scratch under one bf16 declaration. A literal reading
would round the per-step replay scratch to bf16 as well. The implemented
choice keeps the per-step cache fp32 because the delta-rule adjoints
contain (1-beta)-style differences whose bf16 rounding noise is not needed
for the declared tolerance; the sparse checkpoints — the surface where
rounding actually threatens decision consistency — are bf16 as registered,
and the P1-6/G10 trace-following replay protects every discrete decision
from rounding regardless (the would-flip construction receipts this,
section 2 G10). **Receipted bf16-path gradient evidence**: parameter
gradient max-relative error **2.16e-3** (spilled bf16,
`kernel_match.json:spilled_bf16_ckpt.param_grad_max_rel`) and **2.12e-3**
(resident bf16), both far inside the registered **0.08** gradient bound;
input gradients 1.37e-4 (spilled) / 1.35e-4 (resident). **M2-REVIEW
ADJUDICATION (a) — RESOLVED, ACCEPTED AS IMPLEMENTED** (append-only
clarification in REGISTRATION.md): the precision surface is bf16 sparse
checkpoints as the declared rounding surface with an fp32 per-step replay
cache; trace-following replay makes the cache's precision non-decisional;
the receipted bf16-path gradient parity (2.16e-3/2.12e-3 << 0.08) supports
accepting the implementation as-is. No bf16-cache rewrite is required.

## 5. T2 calculation — REDONE against the actual longest registered input

The registration's T2 resolution: "the decay constants are registered so
that a live slot's fade horizon is at least 10x the longest tested word ...
decay is provably inert within the ladder ... cannot be the binding
constraint on any arm". G11 obliges the freeze to redo the calculation
against the actual longest registered input of the freeze.

**The longest registered input of this freeze is the 90,904-token
max-length numerics smoke** (house standard, registration gate 4 — receipted
at exactly 90,904 tokens, `numerics_smoke_max_length.json:requested_length`,
`length_shortfall: 0`). Every other registered input is shorter: the
ladder's longest positive word (9x bin, k=1152) is 2*1152 = 2,304 a/b
tokens (2,306 with SOS/EOS); the longest malformed word (trailing-a at 9x,
a^k b^k a^j, j <= k) is 3*1152 = 3,456 tokens (3,458 with SOS/EOS); the
cap-stress smoke is T=64.

**Equations.** A live slot's strength decays multiplicatively per token:

    sigma_{t+1} = sigma_t * r          with r = decay_rate = 0.999995

and a slot is freed when its strength reaches the free threshold:

    sigma_T <= f  =>  freed           with f = free_threshold = 1e-3

Worst case (a slot allocated with strength 1.0 and decayed every token from
its birth), the fade horizon H solves r^H = f:

    H = ln(f) / ln(r) = ln(1e-3) / ln(0.999995)
      = (-6.907755278982137) / (-5.000012500036375e-06)
      = 1,381,547.60 tokens

(receipt: `numerics_smoke_max_length.json:t2_fade_horizon.horizon_tokens`
= 1381547.6019068586).

**The >= 10x obligation against the actual longest registered input:**

    10 * L_max = 10 * 90,904 = 909,040 tokens
    H = 1,381,548 >= 909,040          ->  SATISFIED (margin 1.52x)

**Ladder-specific check** (the registration's own worked number): at the 9x
bin a positive word is 2,304 tokens, requiring a horizon >= 23,040 tokens;
H = 1,381,548 covers it by ~60x. Decay is therefore provably inert within
the ladder and at every registered input: a deep-k failure cannot be
attributed to cache policy (any decay-driven failure would have to occur
beyond ~1.38M tokens, more than 15x the longest registered input).

## 6. The unbounded cap (T5) — receipted

`tape_unbounded`/`tape_appendonly` cap = **2^17 = 131,072 slots per (batch
element, head)**. Receipt (`eng/fused/receipts/cap_stress_smoke.json`):
fresh zero-update forward+backward at the cap, all gradients finite
(`all_finite: true`), max allocation target 15 < cap (`alloc_below_cap:
true`), peak allocation 8.98 GiB, zero updates. Cap-hit is right-censored,
never a scientific failure (T5).

## 7. The frozen recipe, data contract and arms

**Recipe** (identical for all seven arms; freeze table section 1): AdamW
(0.9, 0.999)/1e-8/1e-4, lr 1e-4 constant, batch 64, clip 1.0, stage budgets
15k/15k/30k, cap 60k, eval cadence 500, seed 0, validation-only
latest-best checkpoint selection.

**Arms** (registration names; card preferences are PREFERENCES only, cards
0–6, never 7; actual card receipted per launch):

| arm | budget B | pop | decay | merge | card pref | role |
|---|---|---|---|---|---|---|
| tape_B0 | 736 | yes | yes | yes | 0 | fixed-size control (B0 = GDN2 state-cell equivalence) |
| tape_B0x4 | 2,944 | yes | yes | yes | 1 | budget sweep |
| tape_B0x16 | 11,776 | yes | yes | yes | 2 | budget sweep |
| tape_unbounded | 131,072 cap (T5) | yes | yes | yes | 3 | the unbounded fork |
| tape_appendonly | 131,072 cap | **NO** | yes | yes | 4 | pop load-bearing control |
| gdn2_native | fixed matrix state (562,834 params at the frozen vocab-5 encoding; M2-review correction F11 — the inherited X27 vocab-8 whole-model label was 563,218; backbone dimensions unchanged, the difference is exactly the shared tied embedding/output weight (8-5)*128 = 384; receipted by construction in `receipts/native_parameter_totals.json`) | — | — | — | 5 | native control |
| e97_native | fixed vector state (569,888 params at the frozen vocab-5 encoding; M2-review correction F11 — the inherited X27 vocab-8 whole-model label was 570,464; backbone dimensions unchanged, the difference is exactly the shared tied embedding/output weight (8-5)*192 = 576; receipted by construction in `receipts/native_parameter_totals.json`) | — | — | — | 6 | native control |

**Data contract** (immutable, seed-fixed; all pool hashes receipted in
`generators/receipts/determinism_check.json`, same-seed byte-identical):

* **Stage 1 — associative recall** (`generators/stage1_mqar.py`): the
  reviewed MQAR family, vocab 64, T=128 train/validation (8 pairs), T=256
  length-band diagnostic (16 pairs), pools 70,000/20,000/10,000, seeds
  28301/28302/28303; metric: mean per-class recall over the 64 classes.
* **Stage 2 — append-necessity**
  (`generators/stage2_append_necessity.py`): the reviewed fresh-key
  selective-copy family, 16 content tokens, T=128, K=4 copied items,
  pools 70,000/20,000, seeds 28401/28402; occupancy-instrumentation grid
  K in {2,4,8,16} at T=256 (256 examples each, seeds 28403+k), never trained
  on; per-example item counts carried for the occupancy receipt.
* **Stage 3 — the ladder** (`generators/stage3_ladder.py`): scalar-token
  convention PAD=0/A=1/B=2/SOS=3/EOS=4 (vocab 5, explicit SOS/EOS
  per the merrill26_separators standard); accept a^k b^k; reject = the
  three X28T malformed shapes equally weighted (leading-b b^k a^k,
  count-mismatch a^j b^k j!=k, trailing-a a^k b^k a^j, j in [1..k]),
  sampled within the pool's depth band; train pool k in [1..128] balanced
  50/50 (70,000, seed 28501); stage-3 validation pool in-band (2,000,
  seed 28502 — the OOD bins are test-only); immutable bins 512 balanced
  words each (seeds 28511–28514); fine sweep pools 256 balanced words per
  k, seed = 28,600 + k (all arms share sweep pools at shared k);
  final-only binary decision at EOS; reporting: binary balanced accuracy +
  per-class recalls + confusion + class counts + majority baseline per
  pool (X26c discipline).
* **Native controls** run stage-3 data with the identical recipe
  (stages 1–2 are tape-ignition scaffolding with no native analogue — the
  declared T6 curriculum confound).

## 8. Package inventory, tests, and the manifest rule

Files (all SHA-256-hashed in `freeze_manifest.json`):

* registration identity: `REGISTRATION.md`, `TASK_DIRECTIVE.md`
* freeze document: `ENGINEERING_FREEZE.md` (this file)
* phase-1/2 code: `eng/{__init__,tape_head,routing_trace,oracle,bptt}.py`,
  `eng/fused/{__init__,fused_layer,tape_kernels,run_gates,run_smokes}.py`,
  both phase-1/2 test suites, and the pure M2-review-correction suite
  `eng/tests/test_trainer_wiring.py` (F1/F2/F5/F7/F8/F9/F10; no CUDA)
* gate/smoke receipts: `eng/fused/receipts/{kernel_match,replay_consistency,
  numerics_smoke_max_length,cap_stress_smoke}.json` + the seven per-arm
  gate-4 smokes `eng/fused/receipts/per_arm_smokes/<arm>.json`
* configs: `configs/{tape_B0,tape_B0x4,tape_B0x16,tape_unbounded,
  tape_appendonly,gdn2_native,e97_native}.json` (each carries its F5
  microbatch binding)
* generators: `generators/{__init__,stage1_mqar,stage2_append_necessity,
  stage3_ladder,check_determinism}.py`,
  `generators/receipts/determinism_check.json`
* phase-4 freeze receipts: `receipts/{microbatch_allocation_model,
  native_parameter_totals}.json`, `build_manifest.py`,
  `make_freeze_receipts.py`
* trainer and launcher: `trainer.py`, `launcher.py`

**Manifest rule**: `freeze_manifest.json` (rebuilt by `build_manifest.py`,
the one spelling) maps every package file above (excluding the manifest
itself) to its SHA-256 — AND, per M2-review correction F4, the
LOAD-BEARING DEPENDENCY IDENTITIES: the `../`-escaped rel paths of
`ndm/models/hybrid_ladder.py`, its transitive native implementation files
used by the two native arms (the E97 triton kernel path and the fla-gdn
path; the GatedDeltaNet implementation itself is the external pip `fla`
package, bound as environment identity), and `scripts/gpu_lease.sh`.
The manifest hash is SHA-256 over the sorted, newline-terminated lines
`<sha256>  <relpath>` (`hash_rule` field). The trainer re-hashes every file
from disk, checks the manifest self-hash and the `--frozen-freeze-hash`
argument, and UNCONDITIONALLY checks the worktree is clean and every bound
byte is committed at HEAD — for the package subtree AND every dependency
identity; changes there invalidate the freeze. Any mismatch fails closed
BEFORE an optimizer exists. `--verify-only` verifies the complete freeze and
exits without constructing a trainer (receipted `torch_imported: false`).
Runtime artifacts (trainer/launcher receipts, run dirs) live OUTSIDE the
package in the sibling `results_x28e_memory_head/` directory (F3), so the
package subtree stays verifiably-clean.

**Tests and determinism at the freeze (post-M2-review-correction state)**:
all suites green — 34 phase-1/2 kernel/routing tests (CUDA) + the pure
M2-review-correction suite `eng/tests/test_trainer_wiring.py`; ALL 66
registered generator pools (the 51-point union of the three bounded
configs' fine-sweep windows + the 15 other registered pools) hash-
deterministic with seed sensitivity computed by actual comparison
(`generators/receipts/determinism_check.json`, byte-stable). Zero optimizer
updates anywhere in the freeze evidence.

## 9. Residual M2-review audit items — flagged, never silent

1. **Precision surface** (section 4): registration gate 3's
   "checkpoints/scratch" grouping vs the implemented bf16-checkpoints +
   fp32-per-step-cache split. Receipted bf16 gradient parity 2.2e-3 <<
   0.08. The review must accept the implemented surface or require a bf16
   per-step cache variant re-gated against 0.08.
2. **`tape_appendonly` pop-disabled mechanism**: the registration's arm
   table binds "pop NO" but not the mechanism; bound here as TRAINER-LEVEL
   zeroed pop strengths (identically 0 < pop_threshold, so the read-path
   POP never fires architecturally; decay + merge retained; the single
   pop-off variable vs `tape_unbounded`). A kernel-level pop-off flag was
   NOT added (single spelling of the frozen kernels); flagged for review.
3. **Model wrapper bindings** (the registration binds the head, not the
   wrapper): scalar-token embedding + two-logit binary head (stage 3) and
   per-stage multi-class heads (stages 1–2) are stage-local; the tape layer
   persists across the curriculum; the optimizer is rebuilt at stage
   transitions to include the stage-local parameters (same frozen
   hyperparameters). Bound at freeze, flagged as such.
4. **Stage-3 validation pool** (2,000 in-band words, seed 28502): required
   by "validation every 500 updates on the arm's stage validation pool"
   while the OOD bins remain test-only; a freeze-time binding, flagged.
5. **Degenerate-band sweep pools**: at a single-k band the count-mismatch
   shape cannot sample j != k inside the band; bound as the alternating
   nearest-mismatch rule j = k+1 / max(k-1, 1) (receipted in pool
   metadata). Flagged.
6. **Native stage budgets**: natives are stage-3-only and receive the
   30,000-update stage-3 slice of the registered budget ladder; bound at
   freeze (the 15k/15k stage-1/2 budgets have no native consumer, T6).
7. **Fused-forward `trace_override`**: not supported on the fused forward
   (trace-following replay lives inside `tape_bwd_kernel` and its
   gate-instrumented form); the eager oracle retains the trace-override
   API for the eager-side gates. Carried from the phase-2 flag, restated.
8. **G8 two-logit head**: pre-M2 smokes used a FIXED seeded projection
   (no learned head exists pre-M2); the frozen trainer uses a learned
   two-logit Linear reading the final layer output. Bound at freeze.

Independent acceptance review required before commit; nothing here is a
launch authorization.

---

## 10. M2-review correction dispositions (phase 4 — the post-review package)

The M2 review BLOCKED the pre-correction package (5xP0, 5xP1, 1xP2, 8
adjudications). All findings are corrected in this package; dispositions and
evidence:

* **F1 (P0, stage-1/2 label mapping):** content labels now map to contiguous
  zero-based classifier indices for the loss (`labels_to_classifier_indices`;
  stage-1 offset 1, stage-2 offset 2) and predictions reverse-map for
  reporting (`classifier_indices_to_labels`). Pure loss/metric unit tests for
  BOTH stages: `eng/tests/test_trainer_wiring.py::
  test_label_mapping_loss_and_metric_both_stages` (no CUDA).
* **F2 (P0, the committed-verification bypass):** the `--require-committed`
  bypass no longer exists in any form; worktree+HEAD verification is
  UNCONDITIONAL on every training entrypoint; `--package-root` is gone —
  verification always targets the ACTUAL EXECUTING package;
  `--verify-only` verifies the complete freeze and exits without
  constructing a trainer or importing any model code (structural receipt:
  `torch_imported: false`; AST test `test_verify_only_path_is_structurally_
  trainer_free`; refusal/exit tests use a committed scratch mirror — the
  project repo itself is never touched).
* **F3 (P0, launcher output dirtying the package):** runtime artifacts moved
  OUTSIDE the frozen package: `--results-root` defaults to the sibling
  house-standard `results_x28e_memory_head/` (created at launch); launcher
  receipts likewise. The package subtree stays verifiably-clean.
* **F4 (P0, unbound dependency identities):** the manifest now binds
  `ndm/models/hybrid_ladder.py`, its transitive native implementation files
  used by the two native arms (the E97 triton kernel path and the fla-gdn
  path), and `scripts/gpu_lease.sh` (rel paths `../../../ndm/...`,
  `../../../scripts/gpu_lease.sh` — the package sits three levels below the
  repo root), each with its own SHA-256; `verify_freeze` re-hashes them from
  disk AND HEAD. FLAG: the gdn2_native arm's GatedDeltaNet implementation
  lives in the EXTERNAL pip `fla` package (site-packages, outside the repo,
  cannot be HEAD-verified); bound as environment identity
  (`freeze_manifest.json:dependency_binding.external_environment_identity`).
* **F5 (P0, gate-4 per-arm feasibility):** resolved by REGISTERED
  EFFECTIVE-BATCH ACCUMULATION (append-only REGISTRATION.md amendment): per
  arm, microbatch budgets bound in `configs/<arm>.json`, allocation-model
  receipt `receipts/microbatch_allocation_model.json` (every in-plan worst
  case keeps >= 8 GiB of the 40-GiB rung), gradient-equivalence test, and
  the seven 90,904-token zero-update per-arm forward/backward/eval smokes at
  the bound budgets (`eng/fused/receipts/per_arm_smokes/`); the cap-stress
  smoke stays for the two cap arms (`cap_stress_smoke.json`, both arms).
* **F6 (P1, censor paths):** unbounded-arm cap hit (R-CAP at the T5 machine
  cap) writes a CENSORED receipt and fails closed (bounded-arm R-CAP merge
  stays normal architecture behavior); nonfinite evaluation logits write a
  CENSORED receipt, never an accuracy number; a projected above-rung single
  example is censored BEFORE any OOM.
* **F7 (P1, occupancy receipts):** peak live slots are measured WITHIN each
  word's length only (to its EOS; PAD-region behavior excluded), occupancy
  at EOS, and OOD occupancy is reported PER-K keyed by depth
  (`per_depth_occupancy`; `meta["depths"]` in the stage-3 pools), feeding the
  registered [0.9k, 1.1k] check.
* **F8 (P1, determinism coverage):** ALL 66 pools (the 51-point union of
  the three bounded configs' windows, derived from the configs — never a
  hardcoded list — plus the 15 other registered pools); seed sensitivity
  computed by ACTUAL hash comparison; the degenerate-band mismatch rule
  receipted in pool metadata AND verified by construction on every sweep
  point; the receipt is byte-stable and rehashed into the manifest.
* **F9 (P1, launcher guards):** TERM->KILL escalation scheduled by an
  INDEPENDENT timer 30s after TERM (not the 60s poll cycle); the escalation
  reaches ANY surviving process in the owned group even if the worker
  identity vanished (group members receipted at TERM time; a recycled
  recorded PID is never signalled); terminal/result/marker receipts carry
  `payload_sha256` and are validated on read — a tampered receipt is
  `error`, never `complete`.
* **F10 (P1, pop-off initialization):** the head is constructed EXACTLY ONCE
  (the `HEAD_CLS` seam in `FusedTapeLayer`; the pop-disabled subclass only
  rewires the forward); unit test
  `test_pop_disabled_shares_identical_non_pop_initial_parameters` proves
  `tape_unbounded` and `tape_appendonly` share identical non-pop initial
  parameters at the same seed.
* **F11 (P2, native totals):** the ACTUAL constructor totals at the frozen
  vocab-5 encoding are receipted BY CONSTRUCTION — 562,834 (gdn2_native) /
  569,888 (e97_native) (`receipts/native_parameter_totals.json`) — and the
  configs/freeze doc distinguish the inherited X27 vocab-8 whole-model
  labels (563,218/570,464; backbone dimensions unchanged; the difference is
  exactly the shared tied embedding/output weight).
* **Adjudications:** (a) accepted as implemented (precision clarification
  above + REGISTRATION.md amendment); (b) accepted with the F10
  initialization fix; (c) mechanism accepted — now operational with the F1
  mapping fix; (d) accepted (unchanged); (e) algorithm accepted —
  receipt/coverage corrected per F8; (f) accepted (unchanged); (g) accepted
  (unchanged); (h) accepted — and the per-arm integration smokes now exist
  (F5).

**Residual flags (never silent):**

1. **Unbounded-arm 9x-bin trailing-a words exceed the 40-GiB rung.** At the
   frozen constants the longest registered input of the two cap arms
   (trailing-a `a^k b^k a^j`, k=1152, j<=k) is 3,458 tokens; its predicted
   peak (fwd+bwd ~47.9 GiB, eval-only ~44 GiB) exceeds the registered
   40-GiB rung. The trainer's fail-closed rung check writes a CENSORED
   receipt for such examples BEFORE any OOM (right-censored infrastructure,
   never a scientific verdict; the registration's "within the machine
   memory budget" conditioning and the OOM right-censoring rule already
   cover this). The balanced/leading-b/count-mismatch 9x words (<= 2,307
   tokens, predicted ~36.5 GiB fwd+bwd; eval-only ~30 GiB) remain within
   the rung. This is a hardware-feasibility bound of the frozen dense-slot
   checkpoint format at CKPT_INTERVAL=16, flagged for the re-review.
2. **The `fla` package** (gdn2_native's GatedDeltaNet implementation) is an
   external pip dependency outside the repo: bound as environment identity
   (version note in the manifest), not HEAD-verifiable. FLAG.
3. **Rung-exceedance censor** (the proactive F6 extension): the registered
   censor list is cap-hit and nonfinite; the projected-above-rung single
   example censor is the fail-closed mechanical form of the registration's
   OOM right-censoring rule, implemented so the censor receipt exists BEFORE
   the process dies. FLAG as an implementation decision for re-review.
4. **FOUND DEFECT (fixed pre-M2): `tape_bwd_kernel` int32 checkpoint-offset
   overflow.** The phase-4 per-arm smokes (F5) exposed that the segment
   replay offsets overflow int32 at the registered store sizes; every
   checkpoint offset in the backward kernel now uses the int64 segment
   index (in-code note: "post-M2-review kernel fix"). All 34 tests and the
   phase-2 gates were re-passed and re-receipted after the fix
   (`kernel_match.json`, `replay_consistency.json`,
   `numerics_smoke_max_length.json`).
5. **FOUND DEFECT — HISTORICAL DISPOSITION, SUPERSEDED (M2-review-2
   finding 9): the diagnosis below was written BEFORE the 09:38
   deterministic-tie-break kernel fix; the CURRENT state is FIXED and
   receipted — the regenerated `tape_unbounded` smoke reports
   `eval_vs_train_output_max_abs = 0.0` (bitwise deterministic), and the
   addendum's item 2 records the fix. The historical
   `tape_unbounded_CENSORED.json` receipt is retained as DIAGNOSTIC-ONLY
   provenance of the pre-fix state and is NOT the arm's current gate-4
   receipt. The original diagnosis follows unchanged:**
   `tape_fwd_kernel` run-to-run VALUE nondeterminism at large spilled
   stores with the pop path active.** On bitwise-identical inputs
   (verified: every scan input hash-identical across fresh processes), the
   `tape_unbounded` forward's read output varies run-to-run by ~2e-6
   (single-sequence probe) to 1.03e-4 (the 88-microbatch smoke max
   eval-vs-train diff). Proven deterministic: all discrete routing
   decisions (case/matched/pop_target/pop_fired/alloc) and the ENTIRE
   final store state; also bitwise deterministic: pop OFF at the same
   2^17 store (`tape_appendonly`) and pop ON at stores <= 16,384 slots
   (the bounded arms); nondeterministic at >= 65,536 slots with pop ON.
   Tested and EXCLUDED: score-tie argmax ordering (decisions never vary),
   software pipelining (`num_stages=1` probe — reverted), allocator
   reuse (fresh processes still wobble). No atomics, no autotune, fixed
   launch config, sequential per-program scan: the residual cause is
   compiler/hardware-level inside `tape_fwd_kernel` and is NOT a quick
   fix. Disposition: the arm's gate-4 smoke receipt stands FAILED
   (`pass: false`) and the fail-closed CENSORED receipt
   `eng/fused/receipts/per_arm_smokes/tape_unbounded_CENSORED.json`
   carries the full evidence; no failure-to-learn is adjudicated; a fix
   requires kernel work plus a full phase-2 gate/test re-validation and
   re-review. FLAG for the re-review.
6. **Smoke-tooling receipt semantics (fixed pre-M2): expected no-grad
   parameters.** `_finite_receipts` in `run_smokes.py` previously recorded
   a parameter with NO gradient as "nonfinite"; under the F10 pop-disabled
   wiring `W_pop` is architecturally unused (pop strengths are identically
   zero constants), so autograd correctly produces no gradient for it.
   The finiteness receipt now distinguishes finite-by-construction
   expected-None gradients (`grad_none_by_construction`, arm-wired) from
   UNEXPECTED None gradients (still a failure); a FAILED per-arm smoke
   additionally writes a fail-closed CENSOR receipt (the F6 discipline
   extended to the gate-4 smokes).

Zero optimizer updates remain: no trainer training path was executed for
any receipt in this package; the per-arm smokes are zero-update
forward/backward/eval only, on leased cards 0-6, NEVER card 7.

## Post-first-M2-review corrections addendum (found and fixed before the re-review)

The first M2 review BLOCKED the package; the correction pass (worker +
parent completion after two worker time-ceiling interruptions) recorded
these additional defects found at ZERO optimizer updates, each with its
fix and receipt:

1. **int32 element-index overflow in `tape_bwd_kernel` checkpoint loads**
   (worker-found, per-arm smoke F5 exposure): ck_values reached 3.12e9
   elements at the B0x16 scale, past 2^31 (B0/B0x4 landed just under).
   Fixed to int64 offsets; all 34 tests and all gates re-passed with
   identical values (`receipts/kernel_match.json` unchanged).
2. **Run-to-run value nondeterminism in the spilled-store fused forward
   with pop ACTIVE on the 2^17 store** (worker probes): root cause
   identical-key score ties broken in tile-reduction order; fixed by
   deterministic tie-breaking in the streaming best-match (`take =
   strictly-greater` with lowest-index-within-tile; earlier tiles win
   ties). Receipted by the regenerated `tape_unbounded.json` smoke:
   `eval_vs_train_output_max_abs = 0.0` (was 1.03e-4). Controls: pop-off
   3 repeats 0.0; bounded store 0.0 (the pre-fix receipts are preserved
   as diagnostics in the review artifacts).
3. **Nonfinite W_pop gradients on the pop-disabled arm** (worker found):
   the pop-kill adjoint was singular at pop_strength = 0. Fixed and
   receipted by `tape_appendonly.json`: pop_fired exactly 0 (the
   single-variable control intact), and W_pop's gradient is correctly
   ABSENT (None by construction — the parameter is architecturally unused
   under the F10 wiring, per finding 6 above); no nonfinite gradient
   appears in any FIRED path. (Wording corrected per M2-review-2 finding
   9: an expected-absent gradient must not be described as a measured
   finite gradient.)
4. **Degenerate-band mismatch rule failed its own k=1 case** (parent
   found): the rule's pass criterion demanded j-alternation at k=1 where
   j=0 is excluded (j never 0), making alternation impossible by design.
   The criterion now requires alternation only when both neighbor bands
   are valid (k >= 2); `mismatch_rule_check(1)` passes with
   `k1_uses_k_plus_1 = True`. Sweep windows (k >= 608) unaffected.
5. **Wiring-test corrections** (parent): the label-mapping metric bound
   test used the wrong averaging convention (registered metric is mean
   per-class recall over classes PRESENT in the labels — the standard
   balanced-accuracy convention); the effective-batch BCE test had a
   per-example-vs-per-position target shape slip; the full-range peak
   assertion contradicted its own trace arithmetic. All test-side fixes
   against implementations verified conformant to the registration.
6. **Determinism receipt byte-stability defect** (parent-found, fixed
   earlier in this pass): wall_seconds timing inside the manifest-covered
   receipt made it fail its own manifest hash on regeneration; timing is
   now stdout-only. (Recorded here for the complete audit trail.)

Per-arm smoke status after all fixes: ALL SEVEN arms PASS (finite, within
the 40-GiB rung, eval-vs-train bitwise 0.0, zero updates, T2 receipted
per arm). The manifest is rebuilt over the final bytes; the fresh
manifest_hash supersedes all earlier hashes.

## M2-review-2 corrections addendum (post-second-review, pre-round-3)

All ten round-2 findings fixed; 54/54 tests green against the final
manifest. (1) Label mapping now runs on the CPU numpy array BEFORE device
transfer (the CUDA-tensor crash is structurally impossible; regression
test present). (2) The training path REQUIRES a validated launcher
context (`X28E_LAUNCHER_LAUNCH_JSON`, arm+freeze-hash checked) and refuses
in-package `--results-root`; `--verify-only` stays independent. (3) The
manifest binds the COMPUTED import closure (136 dependency identities, AST
fixpoint over the hand-listed chain) plus installed-environment identities
(fla 0.5.1 via `flash-linear-attention`, torch version) VALIDATED by
`verify_freeze` via importlib.metadata. (4) Gate 4 restated by append-only
registration amendment: literal single-trajectory smokes receipted where
the frozen budgets admit them (tape_B0, gdn2_native, e97_native:
n_microbatches=1 at 90,904 tokens); aggregate-token smokes at frozen
budgets for the split arms; the two cap arms' literal-trajectory
infeasibility is receipted in `receipts/microbatch_allocation_model.json`.
(5) Exactly-64 sampler with the dropped-remainder epoch-boundary policy
(registered amendment; 70,000-example test present). (6) Native
eval-vs-train diffs are now MEASURED (train-mode capture, eval-mode
compare; both native receipts regenerated, pass with measured 0.0 on
literal trajectories). (7) `arm_state` is identity-bound (arm+freeze
validation, malformed-JSON -> error, bare exit-0 without the completed
result -> error; tests present). (8) A validated identity-matching
`guard.json` is classified as the registered resource-CENSOR (tests
present). (9) The historical nondeterminism disposition is explicitly
SUPERSEDED (diagnostic-only CENSORED receipt); the W_pop wording
distinguishes expected-ABSENT gradients from measured-finite ones. (10)
The native receipt's inherited-label fields now carry the true X27
vocab-8 labels (563,218 / 570,464) with the reconstruction cross-check.

## M2-review-6 corrections addendum (signal-path hardening)

All four round-6 findings fixed, 67/67 tests. (F1) `_escalate` and
`guard_kill` consume ONLY hash-validated, schema-complete receipts at
their own consumption points (`read_worker_record`; the guard's validity
flag honored; malformed guard JSON contained, never re-signed). (F2)
Group-continuity rule: unrecorded group members are signalled ONLY while
the recorded leader is still alive with its receipted starttime; with the
leader gone, `guard_kill` TERMs and `_escalate` KILLs identity-matching
recorded members only, and a reused-PGID replacement group is never
touched. Fail-closed tradeoff, registered here: in the rare
leader-dead-at-TERM case with no prior recorded members, NO member is
signalled (an orphaned child may outlive its receipted run; this leak is
bounded by the child's own lifetime and preferred over signalling a
replacement group's members — collateral kill of foreign processes is the
catastrophic failure mode this rule excludes). (F3) The watchdog
consumes the same validated record as the classifier (full schema incl.
finite numeric `launched_at`); corrupt receipts classify as arm errors and
can never KeyError/TypeError the watcher. (F4) The schema-broken test
fixtures are hash-signed so they exercise the schema branch itself, with
direct `alive()` missing-field assertions.

## M2-review-7 corrections addendum (lifecycle and consumer seams)

All three round-7 findings fixed, 69/69 tests. (F1) The sequential
TERM->KILL lifecycle can no longer promote an unproven group snapshot
into ownership evidence: when leader continuity is absent at TERM time,
`guard_kill` receipts ONLY previously-trusted identity-matching members
into `group_members` (an empty set on a first leaderless guard) — the
30s-later `_escalate` therefore has no replacement-group members to
kill; under proven continuity (leader alive) the current owned-group
snapshot remains trusted as before. (F2) Both signal paths validate the
guard receipt's ARM BINDING and the shape of its recorded members via
`guard_valid_for_signalling` — a checksum-valid wrong-arm guard or a
signed guard with malformed member records produces no signalling and
no re-signing, with a contained failure instead of a KeyError. (F3) The
trainer's entry-context gate checksum-validates the launch receipt
(`payload_sha256` recomputed over the payload) and requires a signed
mapping before checking identity fields — unsigned or stale-checksum
launch evidence is refused at the constructor; the launcher signs its
contexts, so genuine ones always pass.

## M2-review-8 corrections addendum (typed identities and worker binding)

All three round-8 findings fixed, 71/71 tests. (F1) Identity values are
TYPED everywhere they are consumed: `valid_pid` (positive int, booleans
excluded) and `valid_starttime` (digit string) gate guard members in
`guard_valid_for_signalling` and worker records in `read_worker_record`;
`alive` contains ValueError/TypeError from malformed /proc paths. A signed
guard with `pid=[]` members or a signed worker with a null-byte pid is
corrupt evidence: contained, zero signals, guard bytes untouched. (F2)
Worker receipts are ARM- AND FREEZE-BOUND: spawn signs `arm` and
`freeze_manifest_hash` into worker.json, and every consumer (arm_state
classification, the watchdog, and both signal paths) validates them
against the expected launch identity. A checksum-valid cross-arm or
wrong-freeze worker receipt is refused everywhere — the cross-arm kill
hijack is closed. (F3) The trainer identity-refusal fixtures are signed so
they exercise the identity checks past the checksum gate.
