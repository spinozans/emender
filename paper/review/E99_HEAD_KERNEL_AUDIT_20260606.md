# E99 head-kernel & 1.3B-sizing audit (before more CMA)

**Date:** 2026-06-06 · **Task:** `e99-head-kernel-audit` (Evaluator role) · **Type:** audit/repair design, *not* a CMA run.

> **User correction this audit enforces.** Stop treating the current `redo-e99-1-3b`
> run as decisive. Two specific suspicions had to be checked against source and a
> profiler, not asserted: (1) the nonlinear / Emender heads may not be
> high-performance Triton/FLA-native paths, so a wallclock-loss comparison would be
> biased toward native GDN-2 for *implementation* reasons rather than *mechanism*;
> (2) the logged "~1.3B" config is actually ~1.1B, and the head mixture itself
> changes the parameter count, so the mixture search was not size-controlled. Dense
> GDN-2 must remain a baseline/control, **not** the E99 answer by default.

**Scope honored.** No CMA, no 15-min LM probes, no token-matched LM triples, no full
runs, no HF uploads, no `paper/main.typ` edit. The only compute run is tiny
single-batch parameter-counting and a fwd+bwd micro-benchmark, both explicitly
allowed for proving implementation path / perf characteristics. All numbers below
are from the **real** redo implementation in the `redo-e99-1-3b` worktree
(`/home/erikg/ndm/.wg-worktrees/agent-1141`): `ndm/models/typed_head_mixture.py`
(6-type), `ndm/models/gdn2_nonlin_shell.py`, `ndm/models/unified_cell.py`,
`ndm/triton/unified_cell_forward.py`, `ndm/triton/unified_cell_backward.py`,
`ndm/models/ladder_lm.py`, and `experiments/e99_mixture_aware_lm_cma/`.

---

## 0. TL;DR (verdicts)

1. **Two of the three pathways ARE fused Triton; one is NOT.**
   - `gdn2_recall` (native GDN-2): one FLA `chunk_gated_delta_rule` launch over the
     whole sequence — chunk-parallel, tensor-core matmuls. **Fully fused, fastest.**
   - `e97_track / count / latch / nonlin` (UnifiedCell corners): one custom Triton
     kernel `_unified_forward_kernel` whose **time recurrence `for t in range(T)`
     runs *inside* the JIT kernel** (one launch fwd, one bwd). **Genuinely fused.**
     It is a *sequential outer-product scan* (not a chunked matmul), but the
     micro-benchmark shows it is **measured competitive with native GDN-2 (0.93×)** at
     `dim1024/h16` — so the assumed "nonlinear heads are slow" premise is **false**
     for these heads. A scan-vs-chunk gap may open at large `dim`, but it is **not**
     demonstrated and is nowhere near the shell's penalty.
   - `gdn2_nonlin_shell` (the fairness control): a **Python `while` loop over
     `state_chunk=64`-step chunks, each calling `chunk_gated_delta_rule` separately,
     with an eager `phi` between chunks**. The nonlinearity lives **outside the fused
     recurrence**. At `T=2048` that is **32 sequential kernel launches + 31 eager
     maps + a `cat`** per shell sub-block per layer (×17 layers). **NOT a single
     fused kernel.** The per-step capability variant (`state_chunk=1`) is ~2048
     launches → eager-speed.

2. **`gdn2_nonlin_shell` is NOT throughput-fair to native GDN-2.** It is
   mathematically faithful (with `phi=identity` it reproduces native GDN to
   `rel<1e-4`, verified in `tests/test_gdn2_nonlin_shell.py`), which makes it a clean
   *capability/accuracy* control — but it is **not** a throughput-fair head. Any
   "loss-at-wallclock" or "tokens/min" comparison that includes shell heads is
   measuring launch overhead, not the cost of nonlinearity. See §2 + the
   micro-benchmark (§3).

3. **The legacy nonlinear/Emender heads ARE fused AND throughput-competitive at the
   measured shape** (single Triton kernel, not eager Python recurrence; **0.93×**
   native in the micro-benchmark). The naive reading "the nonlinear heads are slow
   eager loops" is **false** for the UnifiedCell corners. The **only** throughput-
   pathological pathway is `gdn2_nonlin_shell` (8.6×–498× slower). So the
   wallclock-bias risk in a mixture CMA comes almost entirely from **including the
   shell control in throughput rankings**, not from the legacy nonlinear heads.

4. **Parameter counting: the logged redo config is 1.103 B, and the head mixture
   moves the count by +32 %.** At the redo's *fixed* shape `dim3328/depth17/
   n_heads102/n_state32` the **only thing that changes is the mixture**, yet total
   params run **1.103 B (dense) → 1.261 B (5:1) → 1.341 B (2:1) → 1.387 B (uniform) →
   1.460 B (all-`nonlin`)** — a **+32 %** swing, with the dense control the
   *smallest* model in the search. So the redo's "fix shape, move mixture" design
   makes the mixture search **also a capacity search**, and "1.3 B" was never
   actually held: dense is −15 % under target. Corrected per-mixture dims that hit the
   1270 M handoff convention / 1.3 B ±2 % are in §4.

5. **Fair next experiment** (§5): either (a) build a **Triton-fused nonlinear-state
   GDN kernel** so all compared heads are single-launch fused, **or** (b) keep the
   shell as an *accuracy-only* control and **report capability separately from
   throughput**, never ranking heads by wallclock-loss across non-fused pathways.
   And **re-derive `dim` per mixture** to hold params at the 1.3 B / 1270 M target so
   the mixture search stops being a size search. A corrected mixture-aware CMA is
   only meaningful **after** one of (a)/(b) and the per-mixture size control is in
   place.

---

## 1. Per-head execution-path table (Q1)

Geometry is matched per head: `head_dim = n_state = 32`, `V = 32`. "Fused" = the
time recurrence executes inside a single kernel launch over the sequence.

| Head type | Source file / class | Recurrence / state update | Kernel backend | Fused? | Expected bottleneck | 1.3B-ready | Fairness verdict |
|---|---|---|---|---|---|---|---|
| `gdn2_recall` | `typed_head_mixture.py` → FLA `GatedDeltaNet(mode='chunk')` → `fla.ops.gated_delta_rule.chunk_gated_delta_rule` | linear gated-delta: `S_t = g_t S_{t-1}(I-β_t k k^T)+β_t v k^T`, read `q_t S_t` | **FLA Triton, chunked** (tensor-core matmuls, intra-chunk parallel, state threaded across chunks in-kernel) | **Yes** (1 launch fwd) | memory BW / matmul; best HW util | Yes (validated path) | **Reference**. Fastest, fully native. |
| `e97_track` | `unified_cell.py:UnifiedCellLayer` (corner `track`, `split_gate=True`) → `ndm.triton.unified_cell_forward._unified_forward_kernel` | split-gated reflection (E97), `phi=tanh` | **Custom Triton**, `for t in range(T)` **inside** the JIT kernel; bwd `_unified_*` kernel with `for ti in range(T)` | **Yes** (1 launch fwd, 1 bwd) | sequential scan over T (no chunk-matmul); **measured 0.93× native** at dim1024/h16 | Yes (fused) | **Throughput-fair (measured 0.93×).** Capability-fair. |
| `count` | same (corner `count`, λ=1, `phi=identity`) | pure integrator `S_t = S_{t-1}+i·outer(k,v)` | same custom Triton scan | **Yes** | sequential scan over T; ~native | Yes | Fair (same fused scan as `e97_track`). |
| `latch` | same (corner `latch`, λ>1, β=0, `phi=tanh`) | bistable ±1 attractor | same custom Triton scan | **Yes** | sequential scan over T; ~native | Yes | Fair (same fused scan). |
| `nonlin` | same (corner `nonlin`, `phi=tanh`/`relu`) | state-nonlinear iterated map `S_t=phi(λS_{t-1}+…)` | same custom Triton scan (`_apply_phi` in-kernel: tanh/relu/gamma-mix) | **Yes** | sequential scan over T; **measured 0.93× native** | Yes | **Fair.** The nonlinearity itself is **in-kernel** — NOT an eager loop, and measured throughput-competitive. |
| `gdn2_nonlin_shell` | `gdn2_nonlin_shell.py:GDN2NonlinShellLayer` → `nonlinear_gated_delta_scan` | native GDN shell + **`S ← phi(S)` at every `state_chunk` boundary** | FLA Triton **per chunk**, wrapped in a **Python `while` loop** (`chunk_gated_delta_rule` called once per 64-step chunk) + eager `phi` + `torch.cat` | **No** (≈`ceil(T/64)`=32 launches at T=2048; `state_chunk=1` ≈ 2048 launches) | **kernel-launch overhead + lost inter-chunk pipelining + eager phi**; not the cost of nonlinearity | Functionally yes; **throughput-unfair** | **Capability/accuracy control ONLY.** Faithful (`phi=identity`≡native, `rel<1e-4`) but **not** throughput-comparable. Do not rank by wallclock-loss. |

Backbone note: the native GDN-2 head can also be sourced from the NVIDIA external
checkout (`ndm/models/external_gdn2.py`, `GatedDeltaNet2`/`chunk_gdn2`), which is
likewise a fused chunked Triton path; the redo LM path uses the FLA `GatedDeltaNet`
above.

---

## 2. Is `gdn2_nonlin_shell` genuinely fused/high-performance? (Q2)

**No — it is a faithful wrapper, not a fused kernel.** Direct from
`gdn2_nonlin_shell.py:nonlinear_gated_delta_scan` (lines 87–107):

```python
while start < T:
    end = min(start + state_chunk, T)
    o_c, S = chunk_gated_delta_rule(q=q[:,start:end], ..., initial_state=S,
                                    output_final_state=True, ...)
    outs.append(o_c)
    if end < T:
        S = _phi(S, state_nonlin)        # eager nonlinearity BETWEEN fused chunks
    start = end
return torch.cat(outs, dim=1)
```

- The **intra-chunk** dynamics use the fused FLA Triton kernel. The **inter-chunk**
  state transition (the nonlinearity) is applied **outside** the fused recurrence, in
  eager PyTorch, at a Python loop granularity of `state_chunk` steps.
- The LM path instantiates the shell with `shell_state_chunk=64`
  (`typed_head_mixture.py:158`). At `T=2048` that is **32 separate
  `chunk_gated_delta_rule` launches + 31 eager `phi` maps + one `cat`** per shell
  sub-block per layer, vs **one** launch for the native head. With `depth=17` and
  shell heads present, the shell pathway alone is ≈`17×32 = 544` sequential launches
  per forward.
- The capability-probe variant `state_chunk=1` is a genuine per-timestep nonlinear
  recurrence but degenerates to **~2048 sequential launches** — effectively eager
  speed.

**Verdict:** the shell "calls native GDN-2 chunks and applies nonlinear state
transforms outside the fused recurrence" — exactly the non-fused case. It is **not
comparable as a throughput-fair head.** It is, however, a *correct* head for
**capability/accuracy** comparison (identity≡native is verified), which is its
intended role as the nonlinearity-isolation control. Keep it for the capability axis;
**exclude it from any tokens/min or loss-at-wallclock ranking.**

---

## 3. Are the legacy nonlinear/Emender heads high-perf at 1.3B? (Q3)

**The UnifiedCell corners (incl. `nonlin`) are fused Triton — not eager.** Evidence:

- `unified_cell.py:UnifiedCellLayer.forward` (line 564) calls `unified_cell(...)`,
  the autograd `Function` wrapping `ndm.triton.unified_cell_forward`.
- `unified_cell_forward.py:_unified_forward_kernel` is `@triton.jit` and carries the
  **time loop `for t in range(T)` inside the kernel** (line 154); the per-step
  nonlinearity is the in-kernel `_apply_phi` (tanh/relu/gamma-mix). One launch fwd.
- `unified_cell_backward.py` is the matching `@triton.jit` bwd with `for ti in
  range(T)` (line 109). One launch bwd.

So the legacy `nonlin` head does **not** use unfused PyTorch/eager recurrence. The
remaining structural difference vs native GDN-2 is **mechanism of fusion**:

- native GDN-2 = **chunked matmul** scan (tensor cores, intra-chunk parallelism);
- UnifiedCell corners = **sequential outer-product scan** (one timestep at a time
  in-kernel, no chunk-level matmul parallelism).

A priori that could bias a wallclock comparison toward native GDN-2 — **but the
micro-benchmark shows it does not, at the measured shape:** the UnifiedCell fused scan
runs at **0.93×** native (slightly faster). The scan-vs-chunk disadvantage may emerge
at larger `dim` where tensor-core matmul throughput dominates, but that is a
*projection*, not a measured fact, and it is small relative to the **only** genuine
implementation artifact, `gdn2_nonlin_shell` (§2), which is **8.6×** slower at the LM
`state_chunk=64` setting.

### Micro-benchmark (fwd+bwd, one GPU, real Triton kernels)

`experiments/e99_head_kernel_audit/microbench.py`, bf16, `B=4, T=2048, dim=1024,
n_heads=16, n_state=32`, 20 timed iters after 3 warmup. **Real numbers
(`microbench.json`):**

| Pathway | ms/iter (fwd+bwd) | slowdown vs native | fused? |
|---|---:|---:|---|
| `native_gdn2_recall` (FLA chunked) | 7.86 | **1.00×** | yes (1 launch) |
| `unified_nonlin_fused` (Triton in-kernel scan) | 7.34 | **0.93×** | yes (1 launch) |
| `shell_nonlin_chunk64` (Python loop, 32 launches) | 67.86 | **8.63×** | no |
| `shell_nonlin_chunk1` (Python loop, ~2048 launches) | 3914.98 | **497.98×** | no |

**The headline is the surprise in row 2:** the legacy fused UnifiedCell nonlinear
scan is **as fast as native GDN-2** (0.93×) at this shape — so the premise "the
nonlinear/Emender heads are slow unfused paths" is **false for the UnifiedCell
corners.** The *only* throughput-pathological head is the shell control: **8.6×**
slower at `state_chunk=64` (LM setting) and **498×** at `state_chunk=1` (capability
setting), purely from kernel-launch overhead + lost inter-chunk pipelining + eager
`phi`. Caveat: at `dim1024/h16` the native chunked-matmul kernel has not yet pulled
away from the sequential scan; at the full `dim3328/h102` LM shape native GDN's
tensor-core advantage will widen the native-vs-unified gap somewhat — but nowhere
near the shell's 8.6×.

Reading: the `slowdown_vs_native` column is the **implementation tax** a
wallclock-matched LM probe would charge each non-native pathway *on top of* any real
mechanism difference. It is exactly the confound this audit exists to flag.

---

## 4. Parameter counting & corrected 1.3B shapes (Q4)

**Counts are from the real `LadderLM(level='typed-gdn2-lm')` built on the redo code
path** (`experiments/e99_head_kernel_audit/param_count.py` /
`param_targeted.py`, `vocab=50281` p50k_base, embeddings tied).

**The logged redo config is 1.103 B, not 1.3 B — and the mixture changes the size.**
At the redo's fixed shape `dim3328 / depth17 / n_heads102 / n_state32`:

| Config at fixed redo shape | Exact params | B | Δ vs 1.3 B |
|---|---:|---:|---:|
| dense GDN-2 (all `gdn2_recall`) | 1,102,926,764 | 1.1029 | **−15.2 %** |
| 5:1 GDN:rest mixture | 1,260,502,994 | 1.2605 | **−3.0 %** |

The UnifiedCell corner heads carry **more** parameters per head than GDN heads (extra
split-gate erase + value-write projections, decay-gate proj), so **as the nonlinear
fraction rises, total params rise ~15 %**. Consequence: the redo's "fix the shape,
move only the mixture" design makes **every mixture a different-size model, with the
dense control the smallest.** A mixture-vs-LM-loss curve under that design conflates
*mixture* with *capacity*. This must be fixed before the comparison is meaningful.

**Full fixed-shape spread + derived 1.3 B shapes (real, `param_targeted.json`):**

*Fixed-shape mixture spread* — `dim3328 / depth17 / n_heads102 / n_state32`, only the
head mixture changes (real builds, `param_targeted.json`):

| Mixture (head-type logits) | Exact params | B | Δ vs 1.3 B |
|---|---:|---:|---:|
| dense GDN-2 | 1,102,926,764 | 1.1029 | −15.16 % |
| 5:1 GDN:rest | 1,260,502,994 | 1.2605 | −3.04 % |
| 2:1 GDN:rest | 1,341,041,956 | 1.3410 | +3.16 % |
| uniform 5-way | 1,386,563,978 | 1.3866 | +6.66 % |
| all-`nonlin` | 1,460,099,008 | 1.4601 | +12.32 % |

**A +32 % parameter swing (1.103 B → 1.460 B) from the mixture alone, at identical
shape.** At fixed `dim3328` the mixture search silently walks the model across a
1.10–1.46 B size range — so "mixture vs LM-loss" is partly "capacity vs LM-loss," and
the dense control is the *smallest* model in the search. This is the size confound the
corrected design must remove.

*Per-mixture dim derived to a fixed param target* (depth17/h102/n32 held; real builds):

| Mixture | derived shape (depth17/h102/n32) | exact params | B | Δ 1270 M | Δ 1.3 B |
|---|---|---:|---:|---:|---:|
| dense → 1270 M | **dim 3840** | 1,272,504,748 | 1.2725 | **+0.20 %** ✓ | −2.12 % |
| 5:1 → 1270 M | **dim 3328** (unchanged) | 1,260,502,994 | 1.2605 | **−0.75 %** ✓ | −3.04 % |
| 2:1 → 1270 M | dim 3264 (range floor; true target ≈ dim 3140) | 1,315,257,700 | 1.3153 | +3.56 % ✗ | +1.17 % |
| dense → 1.3 B | **dim 3904** | 1,293,701,996 | 1.2937 | +1.87 % | **−0.48 %** ✓ |

**Read-off:** holding params at 1270 M, the required `dim` spans **~3140 (2:1) →
3328 (5:1) → 3840 (dense)** — a ~22 % `dim` range *for the same parameter budget*.
That is exactly the capacity the fixed-`dim` redo design was silently handing the
nonlinear-rich mixtures. Note the redo's dense config (dim3328) is **−13 %** under the
1270 M convention; a correct dense control needs **dim 3840** (1270 M) or **dim 3904**
(1.3 B). The 2:1 derive hit my probe-range floor (3264); to land at 1270 M it needs
`dim ≈ 3140` — the driver's `derive_dim` should search down to ~3000. (Source:
`experiments/e99_head_kernel_audit/param_targeted.json`.)

**Tolerance / convention.** Project convention (handoff
`HANDOFF_E97_GDN2_CMAES_20260528.md`) targets **1270 M** at ctx2k
(`cmaes_1270M_ctx2k_baselines`); the controls (`E99_1P3B_LM_CONTROLS.md`) derived
**per-arm** shapes to `--params 1270M` (e.g. dense `fla-gdn` dim2688/depth21/h44;
typed dim3072/depth22/h96). So the established convention is **per-arm dim derivation
to a fixed param target**, *not* a fixed shape. "1.3 B ±2 %" = **1.274–1.326 B**; the
1270 M convention (1.246–1.294 B band at its own ±2 %) sits just under it. I recommend
hitting **1270 M** to stay directly comparable to the handoff baselines and the
controls, and reporting the exact count per mixture.

**Corrected sizing rule (the actual repair):** do **not** fix `dim`. For each proposed
mixture, **derive `dim` (depth/heads/state fixed) to the chosen param target** — the
same `derive_dim` pattern already in `run_capability.py:derive_dim` and the controls
harness — so every candidate is param-matched. Concrete candidate shapes that hit the
target are in the table above.

---

## 5. Fair next-experiment design (Q5)

The comparison is only fair if **either** all compared heads are equally
high-perf/fused, **or** the report structurally separates *computational mechanism*
from *implementation artifact*. Recommended design:

**A. Throughput-fairness — pick one before any wallclock ranking:**

1. **(Preferred, needs kernel work)** Implement a **Triton-fused nonlinear-state GDN
   kernel**: take the FLA chunked gated-delta kernel and inject `phi(S)` at the
   in-kernel chunk-state carry (the same place `unified_cell_forward` applies
   `_apply_phi`), so the nonlinear-in-time update is single-launch fused like native
   GDN-2. This makes shell-vs-native a true mechanism comparison. **This is a
   prerequisite for any wallclock-loss claim about nonlinear heads.** Until it exists,
   the shell is accuracy-only.
2. **(If no kernel work)** Keep `gdn2_nonlin_shell` as an **accuracy/capability-only**
   control and **never** include it in tokens/min or loss-at-wallclock rankings.
   Report nonlinear-head value on **token-matched** loss + capability probes only, and
   print the implementation tax (§3 micro-benchmark) next to every wallclock number so
   no reader mistakes launch overhead for mechanism. For the UnifiedCell corners,
   note they are fused-but-sequential-scan and quote their own slowdown factor.

**B. Size-fairness (independent of A):** re-derive `dim` per mixture to the 1270 M /
1.3 B target (§4) so the mixture search is not a capacity search. Log exact params per
candidate; reject any candidate outside the target ±2 %.

**C. Reporting:** every head/mixture row must carry **(loss token-matched, capability
score, tokens/min, fused?/launches, exact params)** as separate columns. Dense GDN-2
stays a labeled **control**; a conclusion may select it **only** if it also wins/ties
**token-matched** loss **and** the capability axis — never on wallclock-loss alone
while non-fused heads are in the comparison (explicit validation constraint).

**Do we need Triton kernels before CMA is meaningful?** For a **mechanism** claim
about nonlinearity that uses wallclock at all: **yes** (option A.1). For a
**capability + token-matched-loss** claim: **no** — the shell is already faithful for
accuracy, and the UnifiedCell corners are already fused; CMA can proceed on a
token-matched + capability objective with the per-mixture size control (B), provided
the report never ranks by wallclock across the non-fused shell pathway.

---

## 6. Recommended next WG tasks

**These three tasks have been created in the graph by this audit** (`--after`
wired so the CMA cannot dispatch until both fixes land):

1. **`implement-triton-fused`** — "Implement Triton-fused nonlinear-state gated-delta
   kernel (GDN shell)" (code, blocker for any wallclock mechanism claim). Inject `phi`
   at the in-kernel chunk-state carry. Validation: `phi=identity` reproduces native
   GDN to `rel<1e-4` (port `tests/test_gdn2_nonlin_shell.py`); single-launch fused
   (profiler: launches ≈ depth, not depth×⌈T/chunk⌉); fwd+bwd grad-finite; throughput
   within ~1.5× of native GDN on the §3 micro-benchmark (vs current 8.6×).
2. **`fix-e99-mixture`** — "Fix E99 mixture-CMA param sizing" (small): change
   `run_mixture_cma.py`/`screen_worker.py` to derive `dim` per mixture to the
   1270 M / 1.3 B target (reuse `derive_dim`; search `dim` down to ~3000 so rich
   mixtures can hit target), logging exact params per candidate and rejecting outside
   ±2 %. Validation: every emitted candidate's counted params within target ±2 %.
3. **`corrected-e99-1-3b`** — "Corrected E99 1.3B mixture-aware CMA" — **`--after`
   `implement-triton-fused`,`fix-e99-mixture`**, and only after this audit passes
   **and a human approves the run**. Token-matched LM loss + capability probes as the
   objective; wallclock reported but **not** used for selection across non-fused
   heads; dense GDN-2 a control. Honors the idle-GPU-only / no-preempt convention.

The in-progress `redo-e99-1-3b` task was also messaged (msg #6) with the two fairness
findings so its current work can incorporate the throughput-fairness and size-control
corrections immediately.

---

## 7. Uncertainty & blockers (explicit)

- **Micro-benchmark is single-shape** (`dim1024/h16`, one GPU) — it establishes the
  *direction and order of magnitude* of the implementation tax, not the exact tax at
  the full `dim3328/depth17/h102` LM shape. It is deliberately tiny (scope: no LM
  runs). The qualitative verdict (shell = non-fused Python-loop; UnifiedCell = fused
  sequential scan; native = fused chunked) is from **source**, not the timing.
- I did **not** re-profile the full 1.3 B LM step (out of scope: no LM probes). The
  launch-count arithmetic (32 shell launches/layer at `state_chunk=64`, T=2048) is
  exact from source; the resulting end-to-end tok/s penalty at 1.3 B is *projected*,
  not measured here.
- I did **not** assert Triton-native status for any head without source/profiler
  evidence: native GDN-2 fused = FLA source + single call; UnifiedCell fused = in
  `unified_cell_forward.py` JIT kernel; shell non-fused = the Python `while` loop in
  `gdn2_nonlin_shell.py`. All quoted with file/line.
- Param counts are from real model construction (tied embeddings, p50k vocab 50281);
  if the corrected run changes vocab/tokenizer or unties embeddings the absolute
  numbers shift (~167 M embedding block), but the **mixture-dependent +32 % swing**
  and the **1.103 B logged-config** findings are robust to that.

**Artifacts:** `experiments/e99_head_kernel_audit/{param_count.py, param_targeted.py,
microbench.py, param_counts.json, param_targeted.json, microbench.json}`.
