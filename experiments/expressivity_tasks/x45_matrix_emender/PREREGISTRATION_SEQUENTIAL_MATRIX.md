# Pre-registration draft — the SEQUENTIAL matrix arm (per-token decisions and reads)

**Date:** 2026-11-14 · **Status: DRAFT, operator sign-off pending** — this is a NEW
registration (it changes the GRANULARITY of the registered algebra, not the algebra),
written before any probe measurement is claimed. Probe-scale engineering checks (parity
gates, memory, timing) are pre-registration evidence and are in scope now; **no 2700 s
claim is run before the operator signs this document off.** It does not touch
`PREREGISTRATION_MATRIX_EMENDER.md`, which stays the registration of the chunk-granular
arm.

## Why this exists

`RESULT_CHUNK_SEQUENTIALITY.md` measured that the registered chunking is a quality
culprit: the chunk dial (the one knob that interpolates sequentiality without touching
registered semantics) climbs monotonically as chunks shrink — 2.4173 → 2.3287 → 2.2462 →
2.2186 nats/byte at chunk 128 → 8, a 0.199-nat gradient at 20 MB probe exposure — and
the dial cannot reach its limit: at chunk 2 the whole-scan arm OOMs (checkpoint memory
scales with chunks/window; 1024 chunk-entry saves × 1 MB = 1.05 GB per scan at the
registered shape) and MERGE starves 27× from chunk 128 to 8. The natural limit of the
dial is chunk 1 — decisions and reads against the CURRENT state at every token — and the
dial's own gradient says that limit is where the quality is. This registration defines
that limit as a first-class architecture and builds it so it is trainable at the
registered shape.

## The architecture under test

**The registered five-op algebra, evaluated at its natural PER-TOKEN granularity.**
Same operators (APPEND / CORRECT / MERGE / POP / DECAY + the certified read), same
thresholds (`tau_match` 0.45, `tau_merge` 0.85, `tau_pop` 0.10, `tie_break` 1e-6, read
scale 2.0, rank 128, no growth), same values — the only change is WHEN the policy looks:

At every token ``t``, against the CURRENT state ``S_t`` (the state as left by token
``t-1``: post-write, post-decay of the previous token — exactly the state the registered
chunk router sees at a chunk-1 boundary; **the per-token semantics are the chunk-1 limit
of the registered dial, not a new policy**):

1. **Decisions** — the registered selection chain (`schedule.route_chunk`'s logic at
   chunk granularity 1): unit-normalise the incoming payload, cosine against the
   unit-normalised live rows, mask dead rows to −2, subtract the index tie-break, take
   best/second/worst with the first-index rule, threshold into the five ops. APPEND
   takes the FIRST free slot (the chunk-1 limit of the ordinal rule: the k-th needy
   token of a chunk takes the k-th free slot, and at chunk 1 every needy token is the
   only one). MERGE blends the two rows with the registered detached strength weights;
   POP targets the worst row under the registered capacity rule.
2. **The read** — the certified read (`read.py`, unchanged formula:
   `out_t = Σ_j softmax_j(scale·cos(q_t, S_t[j]) over LIVE slots) · ‖S_t[j]‖ · S_t[j]`,
   no renormaliser, all-dead ⇒ exactly zero attention), taken **at every token against
   the state the decisions just saw**. This is the registered read machinery at its
   natural per-token granularity — at chunk 1 the chunk-entry state IS the current
   state, so the K23 §7 chunk-constant-weighting distortion (13 % of read scale,
   median) is zero by construction, not by a new read.
3. **The update** — the scan's registered order, verbatim (`algebra
   .sequential_matrix_scan`): the read, then DECAY (``S ← λ_t S``), then every lane in
   order (``S[j] = erase ? p : S[j] + p``, MERGE = CORRECT target + POP source).

No new operators, no threshold changes, no growth: the policy constants are the
registered `RoutingPolicy` verbatim, and `policy.grow` is refused by this arm (growth is
a chunk-boundary concept; the sequential registration does not carry it).

## The arithmetic spec (the K5A obstacle, resolved by definition)

The registered arm computes its decision similarity with torch/cuBLAS bf16 reductions
whose order is not reproducible from a kernel (the banked probe: ~48 % of entries
bit-equal, 0/1024 payload norms — `K5A_ROUTING_DIFF.md`). That ruled out bit-exact
in-kernel decisions THERE because the eager router was the fixed point. Here the oracle
is NEW, so the registration DEFINES the arithmetic and both implementations implement
the same definition:

* **Every reduction is a balanced binary tree over the reduced axis** (the axis padded
  to its next power of two with exact zeros, which contribute exactly 0): repeatedly
  view the axis as `(n/2, 2)` and sum the pairs, `log2(n)` times. Row norms, payload
  and query norms, decision cosines, read scores, the softmax denominator, the read
  output and its adjoint's reductions all use this one order.
* **All decision and read arithmetic is fp32** (inputs upcast from the autocast dtype
  once, at entry). The registered arm's bf16 similarity is a precision artefact of its
  bmm, not a policy constant; this registration states its own precision and keeps it
  bit-reproducible.
* **Correctly-rounded IEEE ops only**: `sqrt.rn.f32`, `div.rn.f32` (both bit-identical
  to torch's `sqrt`/`/`, verified), and the one transcendental — `exp` in the read
  softmax — is the SAME libdevice function torch's `exp` lowers to (verified bit-exact,
  0/2,000,000 mismatches where Triton's default `tl.exp` mismatches ~80 %).
* **The tie rules are torch's**: first-index argmax/argmin (lowest index wins at equal
  score), exactly what the registered `route_chunk` relies on.

The eager per-token oracle (new, in `k6_sequential_matrix.py`) implements this spec in
torch; the Triton kernels implement the identical definition. **Decisions, payloads,
read weights and reads are therefore bit-exact against the oracle by construction** —
the gate-A bar (0 flips at the K5A lane scale) applies with the oracle as the fixed
point, and the parity tests are `torch.equal`, not floor comparisons.

## The kernels (forward, and the backward that dodges the memory wall)

* **Forward: one persistent whole-window kernel** per (layer, window) scan, the K4
  pattern at whole-scan granularity. One program per (batch·head) row holds the ENTIRE
  store (`rank × d_v` fp32 = 128×128 at the registered shape) in registers and walks
  the window's tokens serially: per token it computes the row norms, the decision
  cosines, the five-op selection, the MERGE blend, the read softmax and the read — all
  in the defined order — then applies DECAY and the lanes with `mul.rn`/`add.rn` (no
  FMA contraction, the x44 §6.2 lesson). It records the decision channel, the per-token
  pre-decision liveness, and the **segment-entry state checkpoints every 64 tokens**.
  One launch per scan; no atomics; bit-deterministic.
* **Backward: decisions are recorded constants** (the standing doctrine — there are no
  decision logits). The continuous channels (read weights, payloads, DECAY gate) carry
  gradient to `query`, `payload`, `decay` and the state. **Decision granularity and
  checkpoint granularity are now independent knobs**: the backward recomputes each
  64-token segment's trajectory from its entry checkpoint by replaying the recorded
  per-token schedule (the banked K4 recompute-over-store kernel, reused verbatim — its
  walk is the forward's walk, so the recomputed states are bit-identical to the ones a
  storing forward would have saved), then a token-parallel prep kernel recomputes the
  per-token read weights and their adjoint coefficients, and a value-blocked
  reverse-walk kernel walks the adjoint. The memory arithmetic, stated:
  **34 MB of segment checkpoints per scan (32 × 1 MB) + one 67 MB segment trajectory
  scratch at a time + 2.6 MB of per-segment adjoint channels**, against the chunk-2
  dial's 1.05 GB of chunk-entry saves per scan — the wall that OOMed the dial does not
  exist here, and microbatch 8 fits the card with room to spare (receipted).
* **Size gates and fallbacks, as in K4/K5**: the forward kernel needs the whole store
  in one program's registers, so `rank × d_v` beyond the register budget (16384 fp32 at
  this stack) is **refused by name** to the eager per-token oracle (the same
  semantics, the parity fixed point, slow — a refusal, never a silent change);
  `policy.grow` is refused outright. The custom-op pair (one op per scan, forward and
  backward, the K5B hosting pattern) captures the whole per-segment backward walk into
  a CUDA graph per mixer slot after its first call, so the sequential-decision cost is
  paid in DEVICE time, not in host dispatch.

## Claim set (registered, pending sign-off)

* **S1 sequentiality.** At matched parameters, matched exposure, matched
  data order and one transferred LR (the gdn2 nomination's, the sweep's own rule),
  the per-token arm's held-out NLL **beats the dial's best banked row** (chunk 8:
  2.2186 nats/byte at 20,022,866 targets) on the same probe protocol. The banked
  sweep gradient predicts it should; falsified if it does not.
* **S2 mechanism.** The op-rate diagnostics recover toward the fresh-state regime:
  MERGE rate per token recovers toward/exceeds the chunk-8 rate (0.0092/token; the
  registered chunk-128 rate is 0.2484), CORRECT holds near the chunk-8 saturation
  (~0.92), and read-weight entropy stays at or below the chunk-8 sharpened level
  (1.546 nats).
* **S3 trainability.** The arm trains at the registered shape within the session
  budget: the projected step tok/s and the honest per-scan walls are reported
  first-class (gate D), including the sequential-decision cost a per-token walk
  cannot avoid.

Single seed (3401, the standing one), stated in every receipt; ~20 MB probe exposure
(the sweep's own matched-exposure rule, its honest caveat inherited verbatim: probe-scale
numbers, one transferred LR, not board-scale claims). The board-scale (2700 s) run is
gated on the operator's sign-off of this document.

## Gates before any probe claim

1. **Gate A — decision and read parity vs the eager oracle**: bit-exact
   (`torch.equal`) on every schedule field, payload lane, read weight and read, on
   ragged/edge fixtures covering all five ops, at the registered shape under the bf16
   autocast regime, and the decision-flip test at the K5A lane bar (the 8.65 M-lane
   registered fixture): **0 flips**.
2. **Gate B — gradient parity vs the oracle's autograd**: floors measured first (the
   K23 discipline: the fp32 oracle against its own fp64 twin, per output), the
   kernel-path gradients within `NOISE_FACTOR` (8×) of them, with each output's
   bit-exactness status stated (the softmax adjoint's cross-axis reductions are
   autograd-ordered on the oracle side, so `dstate`/`dpayload` are expected WITHIN
   floors, not bit-exact — stated, not hidden). The segment-recompute trajectory is
   asserted bit-equal to the forward's own checkpoints.
3. **Gate C — the quality probe** (S1/S2 above): 20,022,866-target matched-exposure
   arm, seed 3401, the sweep's protocol verbatim, compared against the banked sweep
   rows.
4. **Gate D — timing**: block-call walls, the projected step tok/s at the registered
   shape, the graph-capture question (does the K5B graph discipline still hold on
   the value/backward side — answered by measurement), and the honest
   sequential-decision cost statement.

## Honest scope

* This is a **new registration of granularity**, not a contradiction of the chunked
  one: the chunked arm's registration and receipts stand untouched; this arm exists
  because the dial's own gradient says the chunk-1 limit is where its quality is.
* The per-token decision walk cannot be as fast as a chunk-128 whole scan; the cost is
  measured and reported, not optimised away by changing semantics.
* The reduction-order spec makes this arm's arithmetic reproducible to the bit across
  implementations — the property the registered bf16 chain does not have and the reason
  this document defines it rather than inheriting it.
