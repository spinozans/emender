# E97 WITHIN-LAYER heterogeneous-head study — unified VERDICT

**Task:** `e97-synth` · **Date:** 2026-06-08 · read-only synthesis (NO training, NO mocks).
Reconciles the five+ within-layer result docs into one answer: **is the within-layer
heterogeneous-head setup APPROPRIATE — LM-competitive AND expressive — and should we commit
it (full scale run / paper)?**

## Sources reconciled

| doc | what it settles |
|---|---|
| `E97_WITHIN_LAYER_STUDY_RESULTS.md` | the matrix: recall recovery (axis 1) + LM held-out BPB (axis 2), both axes, 7 configs |
| `E97_GENERALIZATION_AUDIT_RESULTS.md` | measurement sanity + held-out re-anchor (train-loss vs held-out ranking flip) |
| `E97_SCALE_PILOT_RESULTS.md` | does the winner hold at 0.48 B, token-matched, on held-out BPB + the throughput cost |
| `E97_WITHIN_LAYER_HEADS_NOTE.md` | the kernel/heads infra: `e97_raw`/`e97_delta` as fused within-layer head types, parity, no-eager-guard |
| `E97_FUSED_LM_KERNEL_NOTE.md` | fused split-edit kernel (43–266× over eager) — the efficiency story |
| *comp-cma findings* | per-head value / optimal composition — **see caveat C6**: `E97_COMP_CMA_RESULTS.md` was never committed to `main` (agent-1207 worktree removed, branch absent). Findings below are reconstructed from the task's logged results and are corroborated by the within-layer matrix; treat as un-rederivable from a committed doc until regenerated. |

**Grounding discipline.** Every LM claim below is anchored on **held-out BPB on the
schedule-free averaged weights**, never raw/live training loss. The audit proved this
matters: in the time-bounded regime, raw train loss ranks cells by *step count*, not quality,
and inverts the top of the table.

---

## Q1 — Does ONE within-layer cell do recall + count + latch + nonlin + track (via gdn-neg)?

**YES — on the expressivity battery this is unambiguous and reproduces at scale.**

The cell is `e97_raw + gdn-neg + MLP`: one `TypedHeadMixtureLayer` per layer holding `e97_raw`
split-edit backbone heads **and** `gdn-neg` (GDN-2 recall head with the negative along-key
eigenvalue) heads **in parallel**, summed into the residual, plus a SwiGLU MLP.

| primitive | `e97_raw` pure | **`e97_raw + gdn-neg`** | mechanism |
|---|---|---|---|
| RECALL (mqar) | 0.14 (blind) | **0.96** | the `gdn-neg` recall head |
| TRACK (s5) | 0.10 (blind) | **1.00** | the *negative eigenvalue* of `gdn-neg` |
| COUNT (anbncn) | 1.00 | **1.00** | the `e97_raw` backbone |
| LATCH (flag-hold) | 1.00 | **1.00** | the `e97_raw` backbone |
| NONLIN (iter-map) | 0.68 | **0.93** | backbone + mix |

- Pure `e97_raw` (and pure `e97_delta`) are **recall-blind and track-blind** (≈ random). Adding
  *any* gdn head lifts recall to ~0.96; only `gdn-neg` *also* drives track to 1.00 — recall and
  state-tracking from one head type. This is the load-bearing finding: the within-layer mix adds
  capability the e97 backbone genuinely cannot express alone.
- **Confirmed at 0.48 B scale** (scale-pilot spot-check): winner recall 0.96 / track 1.00 /
  count 1.00, with track+count **extrapolating to 8× train length** (track 0.85 @ T=1024);
  `e97_raw` control stays recall-blind (0.13) / track-blind (0.11). The property is intrinsic to
  the head composition, not a small-scale artifact.

So: **one cell, all five primitives. The capability claim is real and scale-robust.**

---

## Q2 — Is it LM-competitive on HELD-OUT BPB — at small scale AND at the pilot scale?

The answer depends on **scale and the compute regime**, and this is where the capability story
and the LM story diverge. All numbers are held-out BPB on averaged weights.

### Small scale (159 M, 17.5-min time-bounded screens)

| config | within-layer study (1 seed) | audit2 (2-seed mean, held-out re-anchor) | #prim |
|---|---|---|---|
| `e97_raw + MLP` (raw_none) | **3.231** (best) | **3.300** (#1 both seeds) | 3/5 |
| `e97_raw + gdn` | 3.272 | — | 4/5 |
| **`e97_raw + gdn-neg + MLP`** (winner) | 3.398 (ties ref) | **3.338** (#2, *beats* ref) | **5/5** |
| `gdn2-mlp` ref (pure gdn-neg) | 3.393 | 3.389 (#3) | 5/5 |
| `e97_delta + MLP` | 3.376 | 3.398 (#4) | 3/5 |

**Reading (small scale).** On held-out BPB the capability-complete winner is **competitive**:
2nd place, *beating* the gdn2-mlp recall reference and trailing only the 3/5 pure-`e97_raw`
champion (by ~0.04 BPB). The audit's two robust, both-seed findings: (i) `raw_none` is #1 on
held-out, and (ii) the **train-loss ↔ held-out ranking flips** — gdn2 "wins" on raw train loss
only because it ran 3.6× more steps in the fixed time budget; held-out token-efficiency favours
the e97 backbone. The real train→held generalization gap is ≤0.025 BPB (held-out is
in-distribution); **~98 % of the apparent "gap" is a units + live-vs-averaged-weights artifact.**
*Caveat:* seed spread is ~0.05–0.08 BPB, so the fine ordering of the three trailing cells
(raw_gdnneg / gdn2 / delta) is within noise — what's robust is raw_none #1 and the flip.

### Pilot scale (0.48 B, TOKEN-MATCHED 4000 steps = 24.6 M tokens, param-matched to ~481 M)

This is the decisive measurement — equal tokens, equal params, fused, held-out:

| config | role | **held-out BPB** | wall-clock (4000 steps) | throughput vs gdn2 |
|---|---|---|---|---|
| `gdn2_mlp_ref` | GDN-2 + MLP | **3.3524** | 0.394 h | 1.00× |
| **`raw_gdnneg`** | **within-layer winner** | **3.3636** (+0.0112, +0.33 %) | 1.014 h | **0.39× (≈ 2.6× slower)** |
| `raw_none` | e97_raw + MLP | 3.8951 (−0.53) | 0.917 h | 0.43× |

**Reading (scale).** On the literal head-to-head bar the within-layer cell **MATCHES gdn2-mlp**
(a statistical tie, +0.33 %) and **decisively BEATS e97_raw+MLP** (−0.53 BPB). So it clears
"match/beat both references." **But it buys no BPB advantage** over the simpler gdn2-mlp — the
32 `e97_raw` heads add ~nothing over pure `gdn-neg` (3.364 vs 3.352) while imposing the full
throughput penalty. And the throughput penalty *worsens with scale*: ~0.9× at small dim →
**0.39× (2.6× slower)** at dim 1024, because the split-edit kernel is latency-bound (GPU util
13–15 % vs gdn2's 97 %). At **equal wall-clock** — the regime a real training run lives in —
gdn2-mlp sees 2.6× more tokens and **wins outright** (the cold-cache time-bounded screen showed
this directionally: gdn2 3.375 vs winner 3.470).

**Net:** LM-competitive *on a token-matched bar* (tie at scale, 2nd small-scale), **not**
LM-competitive *on a wall-clock bar at scale* (loses to gdn2-mlp).

---

## Q3 — Recommended architecture + composition + MLP + efficiency cost

**If one commits the within-layer cell at all, it is:**

- **Architecture:** `LadderLM(level='typed-gdn2-lm')` — every layer a within-layer
  `TypedHeadMixtureLayer` (head-type *fractions* in parallel, NOT interleaved whole-layers) + a
  SwiGLU MLP.
- **Composition:** the **2-type** cell `e97_raw + gdn-neg` — a 50/50 split (e.g. 32 `gdn-neg`
  recall heads + 32 `e97_raw` backbone heads at n_heads=64), `gdn_allow_neg_eigval=1`. The 2-type
  cell is the *practical* one: the 6-head-type full-CMA mixture (comp-cma `g2i2`) JIT-compiles
  **six** distinct kernels and never reached a usable step count at scale — prohibitive.
- **MLP:** SwiGLU. `mlp_ratio=1.0` is canonical; at scale, param-match across cells by flexing
  **only** the MLP ratio (a fungible param sink: ≈ 75.5 M params per ratio unit at 0.48 B), which
  leaves the recurrent head composition under test exactly as defined.
- **Per-head value (comp-cma, caveat C6):** only `e97_raw` (LM backbone, −0.34 BPB correlation)
  and `gdn2_recall` heads earn their place. `latch` (+0.87 BPB!), `nonlin` (+0.60), `count`, and
  `e97_delta` are **LM liabilities** the backbone already covers — do not add them as standing LM
  heads. Note the comp-cma LM-best operating point (`g2i2`, 3.203 BPB) used *plain* `gdn` and was
  recall-strong but **track-blind**; capability-completeness *requires* `gdn-neg`, which costs LM.
  So even the optimal composition cannot be both LM-best and 5/5.

- **Efficiency cost:** fusion is essential and works (43–266× over eager; the heterogeneous LM
  sustains ~26 M tok / 18-min screen). The residual cost is **recurrence work, not an eager
  stall**: ~0.27× pure-GDN at 2× params/4 pathways at small dim, degrading to **~0.39× (2.6×
  slower) at 0.48 B** because the split-edit kernel is latency-bound (GPU util 13–15 %). Whether
  this is kernel immaturity or fundamental is **open** — but it is the cost *today*.

---

## Q4 — Honest caveats (what's unproven; where it loses)

- **C1 — No BPB upside at scale.** Token-matched, the winner only *ties* gdn2-mlp; the
  `e97_raw` heads are **dead weight for held-out BPB** over pure `gdn-neg`.
- **C2 — No unique capability.** gdn2-mlp is **itself** capability-complete (recall 0.98 / track
  1.00 / count 1.00 in the within-layer study). The within-layer mix buys no capability gdn2-mlp
  lacks — it reaches the same corner more expensively.
- **C3 — Loses at equal wall-clock at scale.** 2.6× slower → in a real (wall-clock-bounded) run
  gdn2-mlp trains on 2.6× the tokens and wins outright.
- **C4 — The only thing it decisively beats is `e97_raw+MLP`,** which merely confirms `e97_raw`
  *alone* is a poor LM backbone — it does not argue *for* mixing it in.
- **C5 — Capability ≠ LM advantage trade-off is structural.** No single within-layer cell is
  simultaneously the held-out-LM winner *and* 5/5: the LM champ (`e97_raw` pure, 3.231/3.300) is
  recall+track blind; the 5/5 cell (`+gdn-neg`) is the LM-costliest of the e97 cells.
- **C6 — comp-cma doc not committed.** `E97_COMP_CMA_RESULTS.md` is absent from `main` (worktree
  removed); its per-head value claims are reconstructed from logs, not a committed artifact. They
  are corroborated by the matrix (latch/nonlin are solved by every cell ⇒ redundant as heads) but
  should be regenerated before being cited in a paper. *(Follow-up task filed.)*
- **C7 — Seed/measurement noise.** Small-scale held-out spread is 0.05–0.08 BPB; the +0.33 % tie
  at scale is well inside any reasonable noise band. The "tie" is genuinely a tie, not a win.
- **C8 — Unproven escape hatches.** No evidence of (a) an optimized kernel closing the 2.6× gap,
  or (b) a longer-token-budget crossover where the e97 heads start to *help* on BPB. Both would be
  required to flip the verdict; neither is shown.

---

## Q5 — GO / NO-GO

### VERDICT: **NO-GO** for committing a full scale run / paper of the within-layer `e97_raw + gdn-neg` mixture **as an LM backbone.**

The within-layer setup is a **genuine engineering and scientific success** — fused
heterogeneous heads work, parity holds, the loud no-eager guard is armed, and one cell
demonstrably recovers recall + track that pure `e97_raw` cannot express. But as an **LM cell to
commit at scale, it is competitive-but-dominated**, and "appropriate to commit" requires more
than "competitive":

1. **No BPB upside** over gdn2-mlp (tie, token-matched) — C1.
2. **No unique capability** — gdn2-mlp is already capability-complete — C2.
3. **2.6× slower** at scale ⇒ loses at equal wall-clock — C3.
4. A single **mature** kernel (gdn2-mlp) vs three+ split-edit kernels (compile/latency tax).

When a candidate matches the incumbent on quality *and* capability while costing 2.6× the
compute and a more fragile kernel, the incumbent wins. **The data says: plain `gdn2-mlp` (GDN-2
+ MLP) is the better cell at scale.** The e97 split-edit / raw-write heads earn their place on
**targeted expressivity probes (count/latch)**, not as a standing LM backbone.

This is consistent across every doc: the scale pilot's explicit NO-GO, the within-layer study's
"capability and time-bounded LM trade off; no single cell is both LM-best AND 5/5," and the
audit's confirmation that the e97-raw *backbone* advantage is real on held-out but is an advantage
in **token-efficiency at small scale**, not a held-out-BPB win that survives token-matching at
scale.

### Exact next step (NO-GO path)

1. **Adopt `gdn2-mlp` (GDN-2 + MLP) as the scale/production LM cell** — equal held-out BPB,
   equal capability-completeness, ~2.6× throughput, one mature kernel. This is also where the
   parallel E98/E99 line already points (GDN/gated-delta is the recall backbone; exotic
   specialists are a sprinkle, not the backbone).
2. **Reframe the within-layer result as a mechanistic / negative finding, not an LM proposal.**
   The publishable nugget: *within-layer head-type mixing recovers recall+track that a pure
   raw-write (e97) backbone cannot express — but confers no LM advantage over GDN-2, because GDN-2
   already covers those primitives at lower cost.* This is a clean, honest contribution and a
   useful guardrail against "more head types = better."
3. **Keep the fused `e97_raw`/`e97_delta` within-layer head infra** (`typed_head_mixture.py`) — it
   is correct, parity-verified, and the right substrate if the throughput gap is ever closed.
4. **Regenerate `E97_COMP_CMA_RESULTS.md`** (C6) so the per-head-value claims rest on a committed
   artifact before any paper cites them.

### What WOULD flip this to GO (both required, neither shown)

- A **kernel pass** lifting e97 GPU util from ~15 % toward GDN's ~97 % (closing the 2.6×
  throughput gap), **AND**
- A **longer-budget token-matched run** showing `raw_gdnneg < gdn2_mlp_ref` on held-out BPB — a
  *real per-token advantage*, not the tie seen here.

Absent both, committing the within-layer mixture as an LM backbone would spend ~2.6× the compute
for no quality or capability gain over the incumbent.

---

*Synthesis grounded in held-out BPB (not training loss), reconciling all five+ within-layer result
docs. Deliverable for `e97-synth`. `main.typ` untouched.*
