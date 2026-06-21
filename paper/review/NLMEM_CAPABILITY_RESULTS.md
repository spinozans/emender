# nlmem-capability — Capability map of the nonlinear MLP-memory cell vs GDN-2

**Task:** `nlmem-capability` · **Cell:** `mlp-mem` (`ndm/models/mlp_mem_head.py`,
spec `paper/review/NONLIN_MEMORY_SPEC.md`) vs **GDN-2** (gated-delta linear matrix
memory) · **Date:** 2026-06-10
**Env:** NVIDIA RTX 6000 Ada, fp32, torch 2.9.1+cu128, triton 3.5.1. All GPU via the
gpu-broker lease (`scripts/gpu_lease.sh`). REAL synthetic-task generators, REAL training
(schedule-free AdamW, 3 seeds). No mocks.

**The question (spec §8.1):** does a nonlinear memory (read `W2 σ(W1 q)`, nonlinear in
the query) store mappings a *linear* matrix memory (GDN-2 read `S q`, linear in the
query) provably cannot — canonically a **non-bilinear / XOR association**, or a
**mod-k** nonlinear map?

---

## 0. Headline

1. **The as-validated `mlp-mem` kernel is INERT.** Its fused kernel initializes the
   fast-weight state `θ₀ = (W1, W2) = (0, 0)`, which is an **absorbing fixed point** of
   the inner-gradient-descent recurrence: with `W1=0`, `h = tanh(W1·k) = 0` kills the
   `W2` write; with `W2=0`, the back-propagated `δ` kills the `W1` write. So the
   nonlinear memory **never writes anything and reads identically zero** — every token's
   output is `0`. `nlmem-validate` passed only because the kernel and its eager reference
   compute the *same* (zero) function, and the surrounding conv+MLP carried the LM loss
   it reported. **A capability map of this cell as-shipped is a map of an empty mixer.**

2. **Fix:** seed the memory with a nonzero initial fast-weight `θ₀` (a fixed per-head
   `W1_0`, the standard fast-weight/TTT "slow weights"; spec §1.4's optional bias DOF).
   This breaks the fixed point and makes the memory functional. Ported into the **fused
   Triton forward** (`HAS_INIT` path) and validated: fwd `1.1e-6`, all bwd grads `<3e-6`
   vs the eager reference; the zero-init path is unchanged (`rel 0`, still inert).

3. **Capability verdict (functional cell, matched compute):** **HONEST NULL —
   the nonlinear memory unlocks no capability the linear GDN-2 memory cannot reach, and
   is strictly dominated.** GDN-2 solves the whole battery — XOR non-bilinear association
   (1.000), nonlinear mod-k `modular_quadratic` (0.989), recall (1.000), MQAR (0.963),
   parity (1.000); `mlp-mem` ties only on XOR (both 1.000, the gate/MLP supply the
   nonlinearity) and `modular_counter` (both ≈baseline) and loses everywhere else (§4).
   The `mlp_ratio=0` ablation shows GDN-2 still solves XOR without the MLP, via its
   per-layer gate + depth (§5). convergent-loss-null TIE is refuted: the cell is inert
   (loses +2.66 BPB) or unstable (NaNs) as an LM mixer (§6).

---

## 1. The dead-memory discovery (proof)

The `mlp-mem` write is one gated inner-GD step on `ℓ_t(θ)=½‖M_θ(k_t)−v_t‖²` with
`M_θ(x)=W2·tanh(W1·x)` (spec §2). Expanding at `θ_{t-1}=(W1,W2)`:

```
pre = W1·k ;  h = tanh(pre) ;  ŷ = W2·h ;  e = ŷ − v
δ   = (W2ᵀ·e) ⊙ (1−h²)
W2 ← γ·W2 − η·(e·hᵀ)            # rank-1
W1 ← γ·W1 − η·(δ·kᵀ)            # rank-1
```

At `θ₀ = (0,0)`: `h = tanh(0) = 0 ⇒ e·hᵀ = 0 ⇒ W2` stays `0`; and `δ = (0ᵀ·e)⊙1 = 0 ⇒
W1` stays `0`. By induction `θ_t = 0 ∀t`, and the read `out_t = W2·tanh(W1·q_t) = 0`.
**Zero is an absorbing fixed point — the memory is inert for any input.**

- Kernel source: `ndm/triton/mlp_mem_fused.py` lines ~93–94 (`W1 = zeros`, `W2 = zeros`).
- Direct check: a hand-built store→query recall sequence through the eager reference
  reads **norm 0.000** with zero init; **norm > 0** the instant `W1_0 ≠ 0`.
- Regression tests added: `test_zero_init_is_inert_absorbing_fixed_point`,
  `test_nonzero_init_breaks_fixed_point_functional`,
  `test_triton_init_parity_fwd_bwd` (`tests/test_mlp_mem_triton.py`).

**Inert-cell battery (archived, `results_mlp_mem_INERT/`):** the as-shipped cell is at
the random baseline on *every* task — boolean_assoc (XOR) `0.49`, boolean_assoc_lin
`0.53`, assoc_recall `0.07`, mqar_recall `0.017`, modular_counter `0.21`, parity `0.51`
— while the matched GDN-2 baseline solves the recall/association tasks (`1.000`,
mqar `0.93`). The inert mlp-mem also loses the convergent-loss-null LM check by
**+2.66 BPB** at matched tokens (it plateaus at the conv/local-ngram floor, contributing
no recurrent memory). This is the "map of an empty mixer" — superseded by the functional
results below.

## 2. The fix — nonzero initial fast-weights θ₀

`MlpMemHeadLayer(mlp_mem_learn_init=True)` (default) adds a fixed per-head
`W1_0 ~ 0.05·N(0,1)` (`W2_0=0`); the memory now writes from token 0 (`h=tanh(W1_0·k)≠0`).
Threaded into the fused Triton forward as a non-learnable init (the per-head basis is
fixed; the per-sequence inner-GD writes + the outer projection/`η`/`γ` learning do the
adaptation). The functional cell:

- solves linear single-bit recall (`boolean_assoc_lin`) at **1.000** (vs 0.53 inert),
- runs at **fused-Triton speed** (≈0.029 s/step, ~40× faster than the eager fallback),
- matches the eager reference fwd `1.1e-6` / bwd `<3e-6` (`test_triton_init_parity`).

## 3. Method — matched A/B (exact shell, only the cell differs)

Both arms are the **same `MlpMemHeadLayer` shell** (identical FLA-GatedDeltaNet
projections / short-conv / L2-norm / output-gate / `o_norm` / `o_proj`); the only change
is the recurrent cell:

| arm | level | cell | params |
|-----|-------|------|-------:|
| **mlpmem** | `mlp-mem` | nonlinear MLP fast-weight memory (functional, `W1_0≠0`) | 2,296,000 |
| **gdn2** | `gdn-matched` | spec §2.3 degenerate **LINEAR** corner: FLA chunked gated-delta on the identical shell | 2,263,232 |

dim 256, 4 heads, n_state 32, HID 32, depth 4, mlp_ratio 2.0 (fixed O(depth) nonlinear
readout present in BOTH arms), fp32, schedule-free AdamW, 5000 steps, train T=128, 3
seeds. Params match to **+1.45%** (the mlp arm's only extra params are the per-layer
`θ₀` init, `W1_0+W2_0` × depth — the gdn2 baseline is *not* handicapped). Run: `run_mlp_mem_battery.py`; aggregate:
`aggregate_mlp_mem.py`.

**Tasks** (REAL deterministic generators): `boolean_assoc` (in-context XOR of two stored
bits — the **non-bilinear association** separator, §8.1.1), `boolean_assoc_lin`
(matched **linear** single-bit recall control), `mqar_recall` + `assoc_recall` (recall
capacity), `modular_counter` K=5 (**mod-k** linear group control), `modular_quadratic`
(`x←x²+c mod p`, **nonlinear mod-k** separator, §8.1.4), `iterated_nonlinear_map`
(logistic-map state-nonlinearity), `parity` (mod-2 control).

---

## 4. Capability battery — headline (mlp_ratio = 2.0, MLP readout present)

Final eval accuracy at train length T=128 (mean ± std over 3 seeds); gap = mlpmem − gdn2.

| task | baseline | **mlpmem** | **gdn2** | gap | verdict |
|------|---------:|-----------:|---------:|----:|---------|
| `boolean_assoc` (XOR, non-bilinear) | 0.500 | 1.000 ± .000 | 1.000 ± .000 | **+0.000** | **TIE** |
| `boolean_assoc_lin` (linear recall) | 0.500 | 0.667 ± .577 | 1.000 ± .000 | −0.333 | gdn2 |
| `modular_counter` K=5 (mod-k linear) | 0.200 | 0.247 ± .009 | 0.242 ± .010 | +0.005 | TIE (≈baseline) |
| `modular_quadratic` (mod-k nonlinear) | 0.143 | 0.079 ± .001 | **0.989 ± .008** | −0.910 | gdn2 |
| `mqar_recall` (multi-query recall) | 0.016 | 0.000 ± .000 | 0.963 ± .000 | −0.963 | gdn2 |
| `parity` (mod-2) | 0.500 | 0.503 ± .001 | 1.000 ± .000 | −0.497 | gdn2 |

**Reading.** GDN-2 (linear matrix memory) **solves every discriminating task** —
including the **XOR non-bilinear association** (1.000) and the **nonlinear mod-k**
`modular_quadratic` (0.989) the spec predicted the *nonlinear* memory would uniquely
unlock. The nonlinear `mlp-mem` cell **ties only where both arms saturate** (XOR, both
1.000) or where the task is unlearnable by either (`modular_counter`, ≈baseline), and
**loses everywhere else** — it cannot reliably do single-key recall (0.667, seed-fragile),
fails multi-query recall (0.000) and `modular_quadratic` (0.079, *below* baseline), and
cannot track parity (0.503). The nonlinear memory does **not** store any mapping the
linear matrix memory (plus the architecture's gate/depth/MLP nonlinearities) cannot —
the spec §8.1 capability hypothesis is **refuted**.

Wall-clock: `mlpmem` runs at **0.84–0.91× gdn2** (non-chunkable sequential scan vs
chunked gated-delta), so it is also strictly slower — no wall-clock case either.

## 5. Mechanism — the mlp_ratio = 0 ablation (MLP readout removed)

The intended sharp §8.1 test: with the post-head SwiGLU MLP **removed**, does the
functional `mlp-mem` nonlinear read solve XOR where a "linear" GDN-2 cannot?

| task (mlp_ratio=0) | baseline | **mlpmem** | **gdn2** | gap |
|------|---------:|-----------:|---------:|----:|
| `boolean_assoc` (XOR) | 0.500 | 0.333 ± .577 | **1.000 ± .000** | −0.667 |
| `boolean_assoc_lin` | 0.500 | 0.667 ± .577 | 1.000 ± .000 | −0.333 |
| `modular_quadratic` | 0.143 | 0.079 ± .001 | 0.955 ± .010 | −0.875 |

**The ablation does not isolate read-nonlinearity — and the result is decisive anyway.**
Removing the SwiGLU MLP does **not** make GDN-2 a purely linear map: each gated-delta
layer keeps its **SiLU output gate** (`o_norm` gated by `silu(g_proj(x))`), and 4 such
layers compose. So even at `mlp_ratio=0` the GDN-2 arm still solves XOR perfectly
(1.000) and `modular_quadratic` (0.955) — the architecture's gate + depth already supply
all the nonlinearity these tasks need. The functional `mlp-mem` *still loses* in this
regime (XOR 0.333, fragile). **There is no configuration in which the nonlinear MLP-memory
read is load-bearing**: the linear matrix memory, wrapped in the same gated, multi-layer
shell, reaches every capability the nonlinear memory was hypothesized to unlock.

## 6. Convergent-loss-null — matched-token natural-language byte-LM

`scripts/nlmem_convergent_loss_null.py` trains both arms (same shell, exactly matched
params) on the SAME real repo-byte stream for a matched token budget. The spec §8.2
prediction is a **matched-token TIE**. It does **not** hold — in either direction:

- **As-shipped INERT cell:** completes but **LOSES by +2.66 BPB** at matched tokens
  (1.0 M tokens; mlp-mem eval BPB ≈3.68 vs gdn2 ≈1.02, gap re-confirmed at 600/800/2000
  steps and 3 LRs). The inert memory plateaus at the conv/local-n-gram floor and never
  descends; the gdn2 linear memory keeps learning.
- **FUNCTIONAL cell:** **numerically UNSTABLE in LM training** — the fast-weights
  explode and the loss goes non-finite within 25–317 steps across LRs `{3e-4, 1e-3}` and
  T `{128, 256}` (it descends 7.96→3.23 BPB then NaNs). A matched-token BPB cannot be
  measured because the cell diverges. (The T=128 algorithmic battery, under schedule-free
  AdamW, was stable enough to complete; the LM regime under plain AdamW is not.)

So the convergent-loss-null **TIE prediction is refuted**: as an LM mixer the nonlinear
MLP-memory is either **inert** (loses by a wide margin) or **unstable** (NaNs) — never a
tie. This compounds the §4–5 capability null.

---

## 7. Verdict

**HONEST NULL — the nonlinear MLP-memory cell stores no mapping a linear GDN-2
matrix memory cannot, and is strictly dominated at matched compute.**

1. **As shipped (`nlmem-triton`/`nlmem-validate`) the cell is INERT** — its zero `θ₀` is
   an absorbing fixed point, so the memory never writes and reads zero. This had to be
   fixed before any capability question was meaningful (learnable/fixed `W1_0`, ported to
   the fused Triton fwd, validated to ~1e-6).
2. **Even functional, the cell unlocks no capability.** GDN-2 solves the full battery
   including the **XOR non-bilinear association** and **nonlinear mod-k**
   `modular_quadratic`; `mlp-mem` ties only where both saturate (XOR) or both fail
   (`modular_counter`), and loses on recall, MQAR, `modular_quadratic` and parity. The
   gate+depth+MLP nonlinearities of the shared shell already supply everything the
   nonlinear *memory* read was hypothesized to add — confirmed by the `mlp_ratio=0`
   ablation (GDN-2 still solves XOR 1.000).
3. **It is strictly worse**: seed-fragile (recall 2/3 seeds), unstable for LM (NaNs),
   and slower (0.84–0.91× wall-clock, non-chunkable).

The spec's NONLIN_MEMORY_SPEC §8 hypotheses (XOR / capacity / nonlinear-regression /
mod-k separation) and the convergent-loss-null TIE are **refuted with numbers**. This
mirrors every prior exotic head in the program ([[e97-wallclock-cma-shell-flat]],
[[complex-eig-capability-periodic-null]]): the capability the exotic state was meant to
add is already reachable through the architecture's existing pathways.

**Follow-ups worth filing:** (a) port a *learnable* `θ₀` gradient into the fused Triton
backward (the fixed init is seed-fragile); (b) add fast-weight normalization to cure the
LM-training instability — only then could a fully fair LM rematch be run, though the
capability null already makes the case.

## Artifacts

- `ndm/triton/mlp_mem_fused.py` — `W1_0` init threaded into the fused fwd kernel + eager ref
- `ndm/models/mlp_mem_head.py` — `mlp_mem_learn_init` (functional cell) + `cell='gdn'` matched baseline
- `ndm/models/ladder_lm.py` — `mlp-mem`, `gdn-matched`, `gdn-matched-lm` levels
- `experiments/expressivity_tasks/tasks/boolean_assoc.py` — non-bilinear XOR-association task
- `experiments/expressivity_tasks/run_mlp_mem_battery.py` / `aggregate_mlp_mem.py`
- `scripts/nlmem_convergent_loss_null.py` — matched-token LM BPB check
- `tests/test_mlp_mem_triton.py` — inert/functional/Triton-init regression tests
- `results_mlp_mem/` (functional) · `results_mlp_mem_INERT/` (as-shipped inert cell)
