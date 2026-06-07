# E97-CONVERGENT CELL — building & testing the convergent candidate (task e97-convergent)

**Status: IN PROGRESS — methodology frozen, results filling in as the 90-job
expressivity battery + 7-arm LM sweep complete.** Auto-generated result tables are
produced by `experiments/expressivity_tasks/aggregate_e97_convergent.py` and pasted
into the `[auto]` sections below; the prose verdict is written once both axes are in.

## Why this study (supersedes the fragile messages to e97-gdn-hybrid)

Two findings from the e97 line reshaped the target architecture, and this task builds
the actual **convergent-cell** candidate that acts on both rather than hoping the
hybrid agent reads an in-flight message:

1. **e97-DELTA may be the better BACKBONE than e97-raw.** Study A
   (`E97_RAW_EXPRESSIVITY_RESULTS.md`) found e97-delta (delta-correction write,
   `raw_write=0`) strictly dominates e97-raw (`raw_write=1`) on expressivity — it
   solves count + a **length-robust** latch where raw collapses (1.00→0.79), lifts
   nonlin (0.89 vs 0.68) and tracks ~3× better — at only ~0.02–0.06 nats LM cost
   (e97 #3 vs e97-raw #1 on the LM leaderboard).
2. **gdn-neg may give recall AND track from ONE head.** Track needs **negative**
   eigenvalues; `gdn_allow_neg_eigval=1` (gdn-neg) solved S5 at 0.998 while vanilla
   gdn does recall at 0.95. If one fla-gdn head with `allow_neg_eigval=1` covers both,
   the convergent cell needs fewer specialist head types.

This is a **complementary** add (recall is the one capability that is both
LM-relevant AND architecturally absent from e97), not the redundant capability-corner
sprinkle on a GDN backbone that E99 found gets learned away.

## What — a 2×3 matrix, evaluated on BOTH axes

```
backbone ∈ {e97-raw (raw_write=1), e97-delta (raw_write=0)}
   ×  recall-head ∈ {none, gdn (allow_neg=0), gdn-neg (allow_neg=1)}
```

| arm | layer pattern (depth-4 / depth-tiled) | backbone | recall head |
|---|---|---|---|
| raw-none     | E97 E97 E97 E97        | raw_write=1 | — |
| raw-gdn      | E97 fla-gdn E97 fla-gdn | raw_write=1 | gdn, allow_neg=0 |
| raw-gdnneg   | E97 fla-gdn E97 fla-gdn | raw_write=1 | gdn, allow_neg=1 |
| delta-none   | E97 E97 E97 E97        | raw_write=0 | — |
| delta-gdn    | E97 fla-gdn E97 fla-gdn | raw_write=0 | gdn, allow_neg=0 |
| delta-gdnneg | E97 fla-gdn E97 fla-gdn | raw_write=0 | gdn, allow_neg=1 |

`E97` = `E88FLAHybrid(use_split_edit=True)`, state `tanh`. The recall head is the
fused FLA `chunk_gated_delta_rule` (`FLAGatedDeltaNetLayer`); **gdn and gdn-neg are
the SAME fused kernel** differing only in the scalar `allow_neg_eigval` (β×2 →
along-key eigenvalue g(1−β) can go negative). No unfused path, no `gdn2_nonlin_shell`.
Recall heads carry FEWER params than backbone layers, so any capability gained is a
**conservative** result.

### Axis 1 — expressivity battery (`experiments/expressivity_tasks/`)

5 primitives, 3 seeds {42,123,456}, train T=128, eval length-extrapolation
T∈{128,256,512,1024}:

| probe | capability |
|---|---|
| `s5_permutation`         | TRACK (S5 state-tracking, needs negative eigenvalues) |
| `anbncn_viability`       | COUNT (aⁿbⁿcⁿ) |
| `iterated_nonlinear_map` | NONLIN (iterated map) |
| `flag_hold_recall` (K=4) | LATCH (flag-hold, length-robustness) |
| `mqar_recall`            | RECALL (multi-query associative recall) |

Shape: `dim=256 n_heads=32 n_state=32 expansion=1.0 depth=4`, schedule-free AdamW,
lr=3e-4, batch 32, 5000 steps. **bf16 autocast for ALL arms** (E97's bounded tanh
state is faithful in bf16, verified by `parity_e97_bf16.py`; the fla-gdn chunk kernel
rejects fp32 — so the comparison is in one precision). PyTorch reference recurrence
for E97 (NOT `--use_triton_e88`), same as Study A.
Orchestrator: `run_e97_convergent.py` → `results_convergent/e97conv_<probe>__<arm>__seed<seed>.json`.

### Axis 2 — LM loss on REAL Pile (token-matched, bf16)

Reuses sibling C's `lm_hybrid_pile.py` protocol (== the e99 1.3B-controls protocol):
REAL `/home/erikg/elman/data/pile.txt`, p50k_base, chunk 1024, batch 8, **steps 2000**
(equal tokens/step ⇒ matched tokens), held-out on a disjoint seed (7777), BPB by
decoding the exact held-out target tokens back to UTF-8. Shape `dim=512 depth=8
n_heads=8 n_state=64`. Every arm carries **+SwiGLU MLP at `mlp_ratio=1.0`** (study-B
best ratio, `E97_RAW_MLP_RESULTS.md`) wired into `HybridLadderLM`, so the convergent
cells are compared at **MLP parity** with the e97-raw+MLP study-B winner (== the
`raw-none` arm at the same ratio). A `gdn2-ref` arm (typed-gdn2, all-layers, +MLP)
gives the gdn2-mlp study-B baseline at this scale.
Runner: `lm_convergent_pile.py`; orchestrator `run_e97_convergent_lm.py` →
`e97_gdn_hybrid_lm/results_convergent/<arm>_mlp1.0_s0.json`.

NO MOCKS: real Pile tokens, real expressivity generators, real recurrence kernels
(per CLAUDE.md).

---

## [auto] Results

<!-- paste output of aggregate_e97_convergent.py here once runs complete -->
_pending — battery in progress._

---

## Verdict (the three decisions this study must deliver)

_Written once both axes are in. The three questions:_

1. **Does gdn-neg = recall AND track in one head, or trade one for the other?**
   (DECISION A — is any single recall arm high on BOTH mqar and s5.)
2. **Is e97-delta the better backbone?** (DECISION B — delta − raw per capability;
   does the delta backbone fix raw's latch length-extrapolation collapse and lift nonlin.)
3. **Does convergence cost LM loss?** (DECISION C — convergent arms' held-out vs the
   e97-raw+MLP and gdn2-mlp baselines at matched tokens.)

**Winning convergent config:** _NAMED here for the generalization audit + scale._
