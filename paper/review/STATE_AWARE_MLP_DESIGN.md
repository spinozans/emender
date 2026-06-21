# State-Aware MLP for Emender — Design + Rationale

**Task:** `research-state-aware-mlp` (Architect). Research + design: project the recurrent
matrix state into the SwiGLU MLP, **param-bounded and ISO-PARAM** vs the baseline
emender-mlp. Full training A/B is a follow-up implementation task; this doc is the plan +
a real param check.

**Authority for every number here:** `scripts/state_aware_mlp_param_check.py` (closed-form
formula **and** a real `torch` build both reproduce the baseline `1,286,589,072` exactly;
run `python scripts/state_aware_mlp_param_check.py --build`). No mock data — all dims are
read from the real model code or computed from the real geometry.

---

## 0. TL;DR

- **Baseline** emender-mlp = `LadderLM(level='E97')` = per block a single
  `E88FLAHybrid(use_split_edit=True, use_gate=True, head_mix='concat')` mixer + a
  `MixerMLPWrapper` (post-mixer RMSNorm + SwiGLU). dim 1792, depth 11, 216 heads,
  n_state 32, exp 1.0, mlp_ratio 2.262336 → **1,286,589,072 params** (verified).
- The SwiGLU MLP sees **only** the post-`o_proj` 1792-d vector. The 32×32 per-head matrix
  state (1024 numbers/head, 221,184/layer) is contracted by a **single query** to 32
  numbers/head (**3.125%** of each head exposed/step) and then mixed **linearly** by
  `o_proj` (the W_o : 6912→1792). The MLP never sees the matrix and does no nonlinear
  cross-head mixing before the linear collapse.
- **Naive** flatten+project of the full state is 396,361,728 params/layer = **4.36B total =
  3.39× the whole model → infeasible** (confirms the task's framing).
- **Decisive feasibility constraint:** the per-token full state `S_t` is **not available in
  Python** in either path (the fused kernel returns only `(S_final, output)`; eager keeps
  only `Sq` per step + final `S`). The only state signal already exposed in Python is the
  pre-`o_proj` readout tensor (`e88_fla_hybrid.py:1933`).
- **Recommendation:** lead with **M1b_m512** (readout-concat, down-proj 6912→512; **zero
  kernel change**, iso-param-exact, residual −0.098%) paired with **M4a** (Frobenius-norm +
  trace, R=0; cheapest, genuinely nonlinear-in-S, eager-validatable) as the floor/control.
  **Always** run a third **plain-wider-MLP** control at the same budget so a win is
  attributable to the mechanism, not to capacity. Honest prior: **NULL is the most likely
  outcome** (see §7); these two are recommended because they are *cheap to falsify*.

---

## 1. What is thrown away today (real 216-head dims, with code cites)

Per block, the mixer is `E88FLAHybrid` (`ndm/models/e88_fla_hybrid.py`); the post-mixer MLP
is `MixerMLPWrapper`/`SwiGLUMLP` (`ndm/models/ladder_lm.py`).

| quantity | value | where |
|---|---|---|
| key_dim = value_dim = `n_heads*n_state` | 6912 | `e88_fla_hybrid.py:937-940` |
| head_v_dim = `value_dim//n_heads` | 32 | `e88_fla_hybrid.py:940` |
| matrix state `S_h` per head `[n_state × head_v]` | **[32 × 32] = 1024 numbers** | update at `:1877` (`S = tanh(decay*S + outer(delta,k))`) |
| state per layer (×216 heads) | **221,184 numbers** | — |
| query readout `Sq_h = S_h @ q_norm` | **32 numbers/head** = a **rank-1** read | `:1880` (docstring `:43`) |
| → fraction of each head exposed/step | **32/1024 = 3.125%** | — |
| concat over heads (head_mix='concat') | 6912 | `:1933` (`output.reshape(B,T,value_dim)`) |
| `o_proj` (the W_o) | `Linear(6912 → 1792)`, **linear**, no bias | `:1056` (def), `:1934` (apply) |
| MLP input | `norm_2(x + mix_out)` ∈ R^1792 **only** | `ladder_lm.py:1028-1031`; SwiGLU `986-997` |

So: each step, only **one query direction** of each head's 32×32 memory reaches the
readout; the other 31 directions never leave the mixer that step. Cross-head mixing is the
**linear** `o_proj`; the only cross-head nonlinearity is the SwiGLU applied *after* that
linear collapse. The MLP is structurally blind to the matrix state.

## 2. Naive flatten+project blowup (the infeasible baseline)

Flattening `S` (221,184/layer) and projecting to `dim` (1792):
`221,184 × 1792 = 396,361,728` params/layer × 11 = **4,359,979,008 added = 3.39× the entire
1.287B model.** Infeasible — the whole task is to expose state to the MLP *without* this.

## 3. Param-bounded mechanisms (exact added params + iso-param rebalance)

Two families: **(A) re-expose / widen the readout** the MLP already implicitly sees
(M1, M2), and **(B) compute a small nonlinear summary of `S`** the readout cannot
reconstruct (M3, M4). All are injected the same way: produce a per-token summary `s ∈
R^extra_in`, concat to the MLP input so `w1,w2` widen `dim → dim+extra_in` (added MLP =
`2·extra_in·mlp_hidden`); `extra_readout` = params of the summary module `R`.

**Iso-param lever:** shrink the SwiGLU hidden (i.e. `mlp_ratio`) to re-absorb the added
params back to `1,286,589,072`. The table below is emitted by the real param-check script;
every residual is **< 0.4%** (most < 0.15%). `extra_readout` and `extra_in` are **per
layer**; `naive+total` is the added params across all 11 layers *before* rebalancing.

```
mech             +R/layer  +MLP-in    naive+total   iso h iso ratio   resid%
M1a                     0     6912    613,122,048    1152    0.6429   +0.378   (full readout, no down-proj)
M1b_m256        1,769,472      256     42,172,416    3392    1.8929   +0.056
M1b_m512        3,538,944      512     84,344,832    2816    1.5714   -0.098   ← PRIMARY
M1b_m1024       7,077,888     1024    168,689,664    1984    1.1071   +0.112
M2_R2_m512     15,925,248      512    220,594,176     896    0.5000   -0.014   (needs kernel multi-query)
M2_R4_m512     47,775,744      512    570,949,632   MLP-lever INFEASIBLE → shrink dim to 1088
M2_R4_m1024    58,392,576     1024    733,151,232   MLP-lever INFEASIBLE → shrink dim to 896
M3_r8                 512     1728    153,286,144    2432    1.3571   -0.168   (probes; in-kernel)
M3_r16_m512     1,770,496      512     64,891,904    3136    1.7500   +0.141
M3_r32_m512     3,540,992      512     84,367,360    2816    1.5714   -0.096
M4a                     0      432     38,320,128    3456    1.9286   -0.095   ← CONTROL (R=0, norm+trace)
M4b_m256        1,769,472      256     42,172,416    3392    1.8929   +0.056   (per-head row-norms)
M4b_m512        3,538,944      512     84,344,832    2816    1.5714   -0.098
```

Mechanism definitions (≥3 required; 4 families, 12 instances):

- **M1 — Readout-concat / wider pre-W_o readout.** Feed the gated per-head readout concat
  (the 6912 tensor at `:1933`, *before* `o_proj`) into the MLP, optionally down-projected
  (M1a full; M1b 6912→m). *Exposes nothing new about `S`* — the same 6912 numbers `o_proj`
  already consumes; the lever is letting SwiGLU form bilinear cross-head features
  `f(head_i)·g(head_j)` **before** the linear `o_proj` collapse. Cost = `2·extra_in·h`
  (M1a) + a `Linear(6912,m)` (M1b). **Zero kernel change.**
- **M2 — Multi-query readout.** `R>1` queries/head → `R` readout vectors/head, exposing an
  `R`-dim **row**-subspace of each head's state instead of 1 direction. Added = `(R−1)`
  extra query projections `Linear(dim,key_dim)` + a down-proj. Richest *linear* readout but
  the most expensive feasible option; **needs the kernel to emit extra per-token readouts.**
- **M3 — Low-rank fixed bilinear probes.** `r` learned probe pairs `(a_i∈R^32, b_i∈R^32)`
  **shared across heads** → per head `r` scalars `a_iᵀ S_h b_i` (input-independent). Probe
  params `r·(n_state+head_v)=r·64` (tiny). **Only family that exposes information `o_proj`
  provably cannot reconstruct** — bilinear directions of `S` outside the single rank-1
  query. Computed **in-kernel** per step.
- **M4 — State statistics.** Parameter-free nonlinear reductions of `S`: Frobenius norm +
  trace (M4a, 2/head=432) or per-head row-norms (M4b, 6912 down-proj). Nonlinear in `S` →
  adds an invariant outside the **linear** span of the readouts. Computed **in-kernel**
  per step; M4a is the cheapest possible probe (the expressivity floor).

## 4. Feasibility constraint (drives the recommendation)

The **per-token full matrix state `S_t` is not available in Python.** The production fused
Triton/CUDA autograd `Function`s return only `(S_final, output)` — `S_final` is the
**last** state `[B,H,32,32]` and `output` is the **readout** sequence `[B,T,H,32]`
(`e88_fla_hybrid.py:227-228`, `:320-321`); `S` is otherwise only checkpointed every 16
steps for backward, never per token. The eager loop likewise keeps only `Sq` per step and
the final `S` (`:1880-1886`). Consequently:

- **M1 needs zero kernel change** — its tensor already exists at `:1933`.
- **M2/M3/M4 must compute their summary in-kernel** (where `S` is resident in
  registers/SMEM) and emit a small extra per-token output, plus a matching backward VJP
  threaded into reverse-replay BPTT. A bounded but real edit (cf. the nlmem inert-state and
  complex-eig chunked-overflow bug histories), and a throughput risk on the no-NVLink,
  DDP-grad-bound box where 100B feasibility already hinges on throughput (preflight-100b).
  **M4a/M3 are validatable eager-only first** (`S` is materialized at `:1877`); only pay for
  the kernel if eager shows non-noise signal.

## 5. Recommended designs + prototype specs

### Primary — M1b_m512 (readout-concat, down-proj to 512)

Zero kernel change, iso-param-exact, highest adversarially-adjusted score. Honest limit: it
exposes no *new* state info; it tests whether *nonlinear pre-collapse cross-head mixing*
alone moves the needle.

**Iso-param geometry** (param-check table): `extra_in=512`; `R = nn.Linear(6912, 512,
bias=False) = 3,538,944 params/layer`; iso `mlp_hidden = 2816` (down from 4032), eff
`mlp_ratio = 1.5714`; **residual −0.098%**; total stays 1,286,589,072.

**Forward (one block, batch B, time T):**
```
r_cat = output.reshape(B, T, 6912)          # SAME tensor o_proj consumes (e88_fla_hybrid.py:1933)
s     = readout_norm(R(r_cat))              # R: Linear(6912->512, bias=False); RMSNorm(512) REQUIRED
                                            #   (r_cat is post-gate, un-normalized; norm_2 does NOT touch it)
mix_out = o_proj(r_cat)                     # unchanged (:1934) -> residual stream byte-identical to baseline
# MixerMLPWrapper.forward (ladder_lm.py:1028-1031):
u   = norm_2(x + mix_out)                   # [B,T,1792]
z   = cat([u, s], dim=-1)                   # [B,T,2304]
out = mix_out + w3(silu(w1(z)) * w2(z))     # w1,w2: Linear(2304,2816); w3: Linear(2816,1792)
```

**Wiring (exact file:line):**
1. `ndm/models/e88_fla_hybrid.py`
   - `__init__` near `o_proj` (`:1056`): add `self.readout_summary = nn.Linear(value_dim, 512, bias=False)` + `self.readout_norm = RMSNorm(512)`.
   - `forward` at `:1933`: reuse `r_cat`; `s = self.readout_norm(self.readout_summary(r_cat))`.
   - `forward` return `:1955`: `return output, S_list` → `return output, S_list, s`. **Audit every return site** (step-kernel / early-return paths) and the TBPTT/FLA hidden-state threading (`ladder_lm.py:1011`).
2. `ndm/models/ladder_lm.py`
   - `SwiGLUMLP.__init__` (`:989-993`): add `extra_in` arg; `w1,w2 = Linear(dim+extra_in, hidden)`; `w3` unchanged.
   - `MixerMLPWrapper.__init__` (`:1015-1022`): pass `extra_in=512`; pick `mlp_ratio` so `round_mlp_hidden(1792, ratio, 64)` (`:981-983`) = **2816** (1.5714).
   - `MixerMLPWrapper.forward` (`:1028-1031`): unpack `mix_out, h_final, s`; `z = cat([norm_2(x+mix_out), s], -1)`; `mlp_out = self.mlp(z)`.
3. `scripts/state_aware_mlp_param_check.py`: no change — assert the rebuilt model == 1,286,589,072 before training.

### Control / floor — M4a (Frobenius-norm + trace, R=0)

The cheapest mechanism that, unlike M1, carries genuinely **nonlinear-in-S** information.
If M4a shows no eager signal, the "expose more of S nonlinearly" thesis is dead and the
M2/M3 production kernels should not be built.

**Iso-param geometry:** `R=0`; `extra_in = 432` (2/head × 216); iso `mlp_hidden = 3456`
(eff `mlp_ratio 1.9286`, only −14.3% vs 4032 — the smallest shrink of any state-aware
mechanism); **residual −0.095%**; total 1,286,589,072.

**Forward** (per head, per token, from live `S_h∈R^{32×32}` at `:1877`):
```
frob = sqrt(sum_{i,v} S_h[i,v]^2 + eps)     # ||S_h||_F : nonlinear invariant of all 1024 entries
tr   = sum_i S_h[i,i]                        # trace; valid ONLY because n_state==head_v==32 (assert it)
s    = RMSNorm(432)(stack_over_heads([frob, tr]))   # [B,T,432]
z    = cat([norm_2(x+mix_out), s], -1)       # [B,T,2224];  w1,w2: Linear(2224,3456)
```
**Wiring:** validate eager first (compute `frob/tr` from `S` at `:1877`, stack, return as 3rd
value — zero kernel change). Only if eager shows signal, emit `s_t` per token from the
resident 32×32 tile in the fused kernel + backward VJP (`dS += S·(ds/frob)`,
`dS += diag_mask·dtr`) threaded into reverse-replay BPTT.

### Mandatory third arm — plain wider-MLP control

`extra_in=0`, `R` removed, the saved budget re-spent on a **larger** `mlp_hidden` to the
same total. This isolates the mechanism from capacity: without it a tie is uninterpretable
and a regression could just be the MLP-hidden cut (e.g. 4032→2816 for M1b_m512). **Not
optional.**

### Not recommended

- **M2** — most expensive feasible mechanism (+220.6M naive, forcing the only sub-1.0
  `mlp_ratio`=0.5000, gutting ~78% of MLP hidden) for the smallest genuinely-new gain (one
  extra *linear* row-direction). Dominated by M3 (same info class + nonlinear stats at ~¼
  the cost).
- **M1a** — full 6912 into the MLP collapses `mlp_ratio` to 0.6429 (mlp_hidden 1152), a
  brutal cut to the most reliable component; use the down-projected M1b instead.

## 6. A/B protocol (iso-param baseline, offline eval)

**Arms** (all rebuilt to 1,286,589,072 ± <0.4%, verified by the param-check script before any
training; identical recipe to `/mnt/nvme1n1/erikg/ref_emender_mlp/launch_manifest.json` —
schedulefree, lr 0.001007, bf16, chunk 2048, p50k_base, fused Triton):
1. **baseline** emender-mlp (mlp_ratio 2.2623),
2. **M1b_m512** (mlp_ratio 1.5714, R + 512-concat),
3. **M4a** (mlp_ratio 1.9286, norm+trace concat),
4. **plain-wider-MLP control** (mlp_ratio set so the budget matches arm 2/3 with no state
   feature).

**Metric 1 — held-out BPB (offline, from checkpoints; no inline held-out).** Score saved
checkpoints with `scripts/eval_checkpoint.py` (forward-only, CE→BPB on a fixed held-out
tensor) on the **shared** held-out slice
`heldout_pile_tail_p50k_2048_1m.pt` (the disjoint pile-tail tensor already used by the ref
runs), e.g.
`python scripts/eval_checkpoint.py --run-dir <run>/runs --heldout_tensor <...>.pt --out <run>/heldout.csv`.
Compare matched-token **and** matched-wall (the standing emender lesson: token-win often
flips to wall-loss — e97delta-1p3b, e97-lm-1p3b).

**Metric 2 — modular_quadratic length-extrapolation separator** (where emender's
class-separation shows). Use `experiments/grok_symmetric_width/train_grok.py`:
`--task modular_quadratic` (mod 48 per `paper/review/PHI_EXPLORATION_RESULTS.md`), train at
`--seq_len 128`, then its built-in `eval_extrap` sweeps fresh sequences at T ∈
{128,256,512,1024,2048,4096} (the Délétang 16× protocol), logging the `length_extrap`
accuracy dict per T; **3 seeds**, report accuracy at 16× (T=2048). Wire the **same** MLP
variant into the small grok cell (its `--dim/--depth/--n_heads/--n_state/--mlp_ratio` knobs
make the arms iso-config) so the separator measures the MLP change, not scale.

**GO/NULL bar:** a GO requires arm 2 or 3 to beat **both** the baseline **and** the
plain-wider-MLP control on held-out BPB at matched wall **or** to widen the modquad 16×
extrap gap, with the iso-param residual < 0.4% certified by the param-check. Anything else
is a NULL.

## 7. Expressivity expectation + failure modes (honest prior: NULL)

Nearly every emender expressivity lever in this codebase converged to a loss-tie at
iso-param: lb-compare (Emender ties bpb, loses modular_counter), e97delta-1p3b (token-win/wall-loss), opt-synth
(composition NULL), ttt-capability (convergent-null), emender-real-1p3b (NULL on all axes).
These two designs are recommended because they are **cheap to falsify**, not because a win
is expected.

- **M1 NULL by construction:** it exposes zero new information about `S` — the same 6912
  numbers `o_proj` already consumes. If `o_proj`→fat-SwiGLU already approximates the useful
  cross-head products via lifting, M1 is dead. The m512 down-proj `R` is *itself* a linear
  collapse, so the surviving "pre-collapse nonlinearity" acts only on a 512-d linear shadow
  — a real but narrow function-class gain, paid for with a −30% MLP-hidden cut.
- **M4a NULL:** Frobenius is invariant to all of O(32)×O(32), aliasing very different memory
  matrices to identical features and discarding exactly the directional content the query
  captures; with tanh-bounded state (`|S|≤1`), `||S_h||_F ≤ √1024 = 32` encodes mostly
  occupancy/saturation, not fine magnitude. It is 2/1024 per head — the floor.
- **Deeper reason a length-extrap win is unlikely for *any* arm here:** the documented
  modular_quadratic length-extrap separator is **per-step nonlinearity-in-time inside the
  recurrence update** (e97 vs byte-identical e97-lin; grok-symmetric-width / grok-confirm:
  "nonlinearity-in-time is the CAUSAL extrap lever"). M1–M4 change only how the MLP *reads*
  an already-tanh-bounded state; none alter the temporal dynamics. There is no mechanistic
  reason to expect an extrap gain — only a possible LM-BPB or cross-head/occupancy
  capability gain, which the plain-wider-MLP control must isolate.

**Decision rule for the implementer:** if **M4a (eager)** shows no non-noise signal, do
**not** build the M2/M3 production kernels — the "expose more of S" thesis is falsified and
the only surviving lever (M1's nonlinear re-mix) is already covered by M1b_m512. If
**M1b_m512** ties its plain-wider-MLP control, declare the family a NULL consistent with the
standing emender pattern.

---

## Appendix — reproduce the param numbers

```
python scripts/state_aware_mlp_param_check.py            # closed-form (asserts == 1,286,589,072)
python scripts/state_aware_mlp_param_check.py --build     # + real torch build cross-check (both == 1,286,589,072)
```
The script prints: the baseline breakdown, the "what the MLP never sees" accounting, the
naive blowup, and the full mechanism × iso-param table above.

**Provenance:** the §1/§2/§3 numbers and the iso-param table are from this script (validated
against a real build). §5/§7 (recommendation, prototype specs, failure modes) were produced
by a 9-agent design workflow (4 mechanism designers → 4 adversarial fairness/feasibility
verifiers → synthesis), every claim cross-checked against the real code cited inline.
