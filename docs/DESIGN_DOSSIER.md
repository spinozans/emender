# Emender / E88 Design Dossier

Synthesis of the 11 architecture, stability, and systems notes in `docs/`. The
goal is to let a reader (or paper-method-section author) understand the final
E88 architecture without reading all 11 source documents. Each section cites
the upstream doc(s) by short name, and design choices are traced to specific
lines of `ndm/models/e88_fused.py` and `ndm/triton/e88_triton_*.py` where the
choice is materially expressed in code.

Short-name → file mapping:

| Short name | Source file |
|---|---|
| E63 | `docs/E63_NONLINEAR_DELTA_DESIGN.md` |
| ABL | `docs/E88_ABLATION_NOTES.md` |
| BAL | `docs/E88_BALANCED_CONFIG_GUIDE.md` |
| FRO | `docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md` |
| M2C | `docs/M2RNN_E88_COMPARISON.md` |
| TSPEC | `docs/MATRIX_ELMAN_TRITON_SPEC.md` |
| MSE | `docs/MATRIX_STATE_ELMAN.md` |
| MENU | `docs/NDM_ARCHITECTURE_MENU.md` |
| SFD | `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md` |
| STAB | `docs/STABILITY_AND_PARAMETERIZATION.md` |
| FIX | `docs/STABILITY_FIX_GUIDANCE.md` |

---

## 1. Architecture summary - Emender nonlinear delta memory

The public model family is named **Emender**; E88 is the current production
instance. The update mechanism is a many-head nonlinear delta memory,
historically called **Nonlinear Delta Memory** (NDM) in older architecture
notes and some source identifiers. The phrase "matrix-state RNN" is too generic
because M2RNN occupies it.

### Per-token, per-head update (E88)

Each head owns a small bounded associative memory `S ∈ R^{N×V}` (production:
N=V=32). For input projections `k, q ∈ R^N`, `v ∈ R^V`, scalar `decay ∈ (0,1)`
and gate `g ∈ R^V`:

```
k ← k / ‖k‖₂              # L2 normalization
q ← q / ‖q‖₂
retrieved = Sᵀ k          # what memory predicts at address k
delta     = v − retrieved # error
S         = tanh( decay · S + outer(k, delta) )   # bounded delta write
y         = Sᵀ q          # read at q
y         = silu(g) · y   # output gate (write is NOT gated)
```

(MENU §E88 / Emender Abstract View; matches `e88_fused.py:298-316` PyTorch
fallback, line-for-line with the fused CUDA / Triton path.)

### Why this is the chosen update

1. **Nonlinearity inside the recurrence (tanh on S) is the UTM-class lever.**
   Linear-in-h recurrences (E61/E62, DeltaNet, Mamba2 in their core update)
   can only compute regular languages in the limit (E63 §Problem with
   E61/E62). Siegelmann–Sontag requires nonlinear `f(h_{t-1}, x_t)` for
   Turing completeness. The Emender update places that nonlinearity on the
   matrix-valued state itself — `tanh` at `e88_fused.py:308` — rather than
   only on the key or the output.
2. **Delta-rule write (`v − Sᵀk`) makes the memory error-correcting**
   (MENU §E88 / Emender Abstract View). Raw outer-product writes
   (`outer(k, v)`) accumulate without correction, and the M2RNN paper-shape
   uses `tanh(H W + k vᵀ)` — see §5. The delta correction is what lets E88
   learn key–value binding *and* rewrite it when retrieval is wrong, which
   is the abstract copying mechanism the model needs.
3. **Many small heads, not one large matrix.** Production E88 uses 370 heads
   of 32×32 (1.27B run, MENU §E88/Emender Abstract View; BAL §Recommended
   Configs has the 64×32 / 32×64 family at ≤500M). Per-head L2-normalized
   q/k give "many independent addressing programs" — M2C §Working
   hypothesis identifies the paper-shape failure (shared q/k across
   hundreds of heads) as the gradient-conditioning failure mode that E88
   avoids by construction.
4. **Output gating, no write gating.** Production uses `silu(g)` on the
   read output but `use_write_gate=0` (MENU §Current E88 Gate State).
   Bounding the write was tried as `β · k·deltaᵀ` and either tied or hurt
   (MENU §Write Scale / Write Gate; ABL Round 2 — `use_gate=False` slightly
   beat baseline at small scale but later 500M ablations reversed this).
5. **Bounded state via tanh, bounded write amplitude via L2-normalized k.**
   This is the synthesis of the two stability lessons that produced E88 —
   see §3.

### Block-level wrapping (LM)

E88 layers are wrapped in a prenorm LadderLM-style residual stack
(`e88_fused.py:425-429`):

```
h = h + E88Layer( RMSNorm(h) )       # repeat × depth
h = RMSNorm(h)                       # final norm
logits = h @ Embed.weightᵀ           # tied LM head
```

There is **no output RMSNorm inside the E88 layer** — that was removed in
ABL Round 2 (−0.10 nats). The two `RMSNorm`s in the file (`e88_fused.py:382,
384`) are the prenorm and final norm of the LM wrapper, not in the
recurrence path.

---

## 2. Parameterization decisions

| Component | Decision | Source | Code reference |
|---|---|---|---|
| Decay shape | Input-dependent (Mamba2-style): `decay = exp(−exp(A_log) · softplus(α(x) + dt_bias))` | STAB §1 Log-Space Parameterization; ABL Round 1 ("Mamba2-style decay is important; simple sigmoid hurts") | `e88_fused.py:244-248` |
| `A_log` init | `log(U[0, 16])` per-head, in log-space so `exp(A_log)` is always positive | STAB §1 | `e88_fused.py:179-181` |
| `dt_bias` init | Log-uniform on `[0.001, 0.1]`, stored in inverse-softplus space | STAB §2 | `e88_fused.py:184-191` |
| Weight-decay exemptions | `A_log._no_weight_decay = True`; `dt_bias._no_weight_decay = True` | STAB §3 (Mamba2/FLA-GDN convention) | `e88_fused.py:181, 192` |
| Decay precision | float32 intermediates before cast back to `x.dtype` | STAB §1 ("compute in fp32, store in bf16") | `e88_fused.py:247-248` |
| q/k L2 normalization | Always on (`use_l2_norm=True`); bounds write amplitude per head | STAB §4 ("FLA-GDN normalizes keys; critical for delta rule"); ABL Round 2 (`E88b_nol2` → NaN); MENU §Q/K Normalization | `e88_fused.py:240-242`; fused into Triton kernel via `NORMALIZE_KQ` flag at `e88_triton_forward.py:108, 193-199` |
| SiLU on q/k/v | Applied to projections *before* L2-norm, mirroring `E88FLAHybrid` | ABL Round 2 (`E88b_nosilu` Δ=+0.307; `E88b_nosilu_nol2` → NaN) | `e88_fused.py:233-237`; Triton flag `APPLY_SILU_QKV` at `e88_triton_forward.py:109, 184-187` |
| State nonlinearity | `tanh(S)` on the post-write state | E63 §What we need for UTM; MENU §State Nonlinearity (tanh kept "for the theoretical hinge"; ABL §Components That Matter) | `e88_fused.py:308` (PyTorch fallback) and `e88_triton_forward.py:210` (numerically stable via `2·sigmoid(2·pre) − 1`) |
| Output gate | `silu(g)` post-read, no gate on the write path | MENU §Current E88 Gate State; ABL Round 1 (`E88a_nogate` Δ=−0.014); later 500M evidence inverted but production keeps `use_gate=True` | `e88_fused.py:314-316`; fused at `e88_triton_forward.py:219-227` |
| Output RMSNorm inside layer | **Removed** | ABL Round 2 (`E88b_nonorm` Δ=−0.100, biggest single win) | not present in `e88_fused.py` — only block-level `RMSNorm` at `382, 384` |
| Short convolutions | **Removed** | ABL Round 1 (`E88a_noconv` Δ=−0.027) | not present in `e88_fused.py` |
| Head/state geometry | `n_heads × n_state ≈ dim` (projection ratio ≈ 1.0); `n_state = 32` is the production default | BAL §The Balancing Principle; ABL Round 3 (32 heads × 16 / 24×24 → NaN ⇒ keep `n_state ≥ 32`) | constructor defaults `e88_fused.py:146-153` (note: file default `n_heads=104, n_state=32`; production 1.27B uses 370×32 per MENU) |
| Checkpoint interval | Every 16 steps; same as register-owned CUDA kernel | (systems) | `e88_fused.py:33, 152`; mirrored in Triton at `e88_triton_forward.py:68` |

---

## 3. Stability lessons

Three families of failure mode appear across the stability docs. Each entry
gives the symptom, the cause as the docs explain it, and the fix that
shipped in E88.

| # | Failure mode | Root cause | Fix in E88 | Source |
|---|---|---|---|---|
| 1 | E75 (matrix-state delta predecessor) trains with gradient spikes up to ≈344; larger `n_state` → worse loss (anti-scaling) | Outer product `outer(δ, k)` unbounded before `tanh`; saturates the bound or runs through the linear regime unstably | L2-normalize `k` *before* the outer product, so the write contribution is bounded per step | STAB §The E75 Paradox + §Issue 1; `e88_fused.py:240-242` |
| 2 | Direct sigmoid-parameterized decay drifts into unstable regions; gradients explode when decay → 1 | Sigmoid gates have vanishing-gradient extremes and no fine-grained control over decay magnitude | Mamba2 log-space `A_log` + softplus-of-`dt_bias` parameterization, with weight-decay exemption | STAB §1 + §2 + §3; `e88_fused.py:178-192, 244-248` |
| 3 | `E88b_nol2` (L2 disabled) and `E88b_nosilu_nol2` both went NaN within ablation runs | Same as #1 mechanism, expressed during real training; the L2 + SiLU on q/k jointly bound the input scale into the recurrence | Both kept on by default; SiLU(q,k,v) precedes L2-norm | ABL Round 2; `e88_fused.py:233-242` |
| 4 | M2RNN paper-shape (shared q/k across hundreds of value heads) blew up at 1.27B with grad norms 10⁶–10⁷ at LR 1e-4 to 2e-4 | Gradients from many value/forget/gate heads collapse through a single q/k addressing path; geometry is gradient-ill-conditioned even though forward values are tanh-bounded | E88 uses *per-head* q, k, v, decay, and gate; many independent normalized address programs | M2C §Working hypothesis; MENU §Head Organization; geometry implicit in `e88_fused.py:171-176, 229-231` |
| 5 | E88 ablations with `n_state < 32` (e.g. `h32n16`, `h24n24`) went NaN | Heads too narrow; the per-head matrix collapses before the L2-bound is meaningful | Production keeps `n_state ≥ 32`; recommended configs all use 32 or 64 | ABL Round 3; BAL §Recommended Configs |
| 6 | Pre-Level-6 / log_0 polynomial-gated experiments: only the *output* was bounded (`compete × silu`), the recurrent path was not | Bounded output gradients do not constrain dh/dh_{t-1}; recurrent gradients compounded through time | Emender bounds the *state* itself (`tanh(S)`), not only the output. This is the lesson E88 inherits from the failed Level 6 / log_0 architectures | FIX §Problem Diagnosis + §Key Insight; `e88_fused.py:308`, `e88_triton_forward.py:210` |
| 7 | In Triton, the raw `(exp − exp⁻¹)/(exp + exp⁻¹)` formula for tanh overflows around `pre > 44` in fp32 → NaN | exp overflow → inf/inf | Replaced with `2·sigmoid(2·pre) − 1` which saturates instead of overflowing | comment block at `e88_triton_forward.py:208-210` |

Of these, #1, #2, #6 are the three "fundamental" stability lessons the
paper-method section should call out — they map directly to the three
Emender design pillars (L2-bounded write, log-space decay, tanh-bounded
state).

---

## 4. Systems summary

### 4.1 Kernel layout (Triton, ROCm-portable)

Forward kernel (`ndm/triton/e88_triton_forward.py`):

* Layout `[T, B, H, *]` internally; the wrapper transposes at the boundary
  without `.contiguous()` to save ~100 MB/tensor at production scale
  (`e88_triton_optimized.py:58-67`).
* One program per `(batch, head_block)`; `BLOCK_H ∈ {1,2,4,8,16}` is
  autotuned because at `H ≥ ~256` per-program-per-head launch overhead
  dominates (`e88_triton_forward.py:46-55, 281-291`).
* State tile `[BLOCK_H, BLOCK_N, BLOCK_V]` lives in registers / SRAM. Caps
  BLOCK_H by `N·V` to avoid spilling at `N=V=64` (`e88_triton_forward.py:284-291`).
* **Fused into the kernel** along with the recurrence: SiLU on q/k/v
  (`APPLY_SILU_QKV`, `e88_triton_forward.py:184-187`), L2-norm on q/k
  (`NORMALIZE_KQ`, lines 193-199), and the output gate `silu(g) · Sᵀq`
  (`APPLY_GATE`, lines 219-227). Each fusion removes 1–2 PyTorch kernel
  launches per layer call; aggregate saving at depth=14 production is
  ~50–60 ms/step (comment at lines 189-192).
* Numerically stable `tanh` via `2·sigmoid(2·pre)−1` (line 210) — see
  stability lesson #7.

Backward kernel (`ndm/triton/e88_triton_backward.py`):

* **Sparse checkpoint replay.** Forward saves S only every
  `CKPT_INTERVAL = 16` steps (matches the CUDA register-owned kernel,
  `e88_triton_forward.py:65-68`). Backward processes one segment at a time,
  forward-replays the K steps to rebuild per-step `S_{t-1}`, then walks
  backward to apply the chain rule (`e88_triton_backward.py:1-56`).
* Memory shrink: `T/K + 1` checkpoint slots instead of `T` (~16× at K=16),
  documented at `e88_triton_forward.py:24-44`.
* Backward chain rule for the delta-rule write is given explicitly in the
  module docstring (`e88_triton_backward.py:19-40`).

### 4.2 Autograd path

PyTorch sees one `autograd.Function` per layer
(`e88_fused.py:36-127` for the CUDA path; the Triton path is wrapped by
`e88_triton_optimized_apply` at `ndm/triton/e88_triton_optimized.py:25-95`).
The forward returns `(S_final, output)` and stashes
`(k, v, q, decay, g, S_cache)` for backward (`e88_fused.py:74`); the
backward dispatches to `e88_register_owned_backward` when `n_state ≤ 32`
and `head_v_dim ≤ 32` (5–6× faster: 1.5 ms vs 10 ms at 32×32) and falls
back to `e88_fused_backward` otherwise (`e88_fused.py:111-125`).

### 4.3 Why Triton, not HIP

FRO §Port strategy lays out the decision: Triton on ROCm runs the same
source on CUDA + ROCm at ~1.5× the perf of hand-tuned HIP, in ~1 week of
porting work instead of 3–6 weeks. The throughput loss (~30%) is
acceptable to keep one codebase across NVIDIA and Frontier. The
step-kernel Triton prototype that started this work is the file
`elman/models/e88_step_kernel.py` (FRO §Port strategy), now grown into
the full forward + backward in `ndm/triton/`.

A separate matrix-Elman Triton design (TSPEC) had been written for the
E70–E73 variants — outer-product + decay + tanh fused as one kernel,
matrix-vector multiply for read, self-gating fused with matmul output
(TSPEC §Key Operations to Optimize). E88's Triton kernel inherits this
fusion pattern even though E70–E73 themselves were superseded.

### 4.4 Distributed training plan (Frontier)

The two distributed docs lay out a two-stage plan with different
assumptions:

1. **Default plan (assuming ParaRNN does not work for matrix state):**
   ScheduleFree-AdamW per-island + hierarchical local-SGD model averaging
   (SFD §Recommended Training Shape). Per-island = 1 node = 8 GCDs with
   intra-island DDP; inter-island sync every K∈{100, 250, 500, 1000}
   local steps. The 4-GPU 75M smoke established that *model-only*
   averaging (no optimizer-state sync) ties optimizer-state sync by 5K
   steps and is faster (SFD §Local Smoke Results); the 1.27B scale-matched
   smoke established H=250 as the recommended setting (final drift
   0.275 vs 0.123 at H=100; SFD §Scale-Matched E88 Smoke). Outer DiLoCo
   momentum stays at β=0 initially.
2. **High-risk/high-reward path:** ParaRNN (Apple, Oct 2025, arXiv
   2510.21450) parallelizes nonlinear RNNs across the time axis via
   Newton's method on a block-bidiagonal Jacobian. *If* it converges on
   E88's `tanh(decay·S + outer(k, v−Sᵀk))` recurrence at matrix-state
   block size, sequence parallelism becomes available and the whole
   distributed plan changes (FRO §Headline). Convergence is untested;
   the FRO doc allocates one 64×24 Frontier run to a prototype.

Other Frontier-specific facts that affect the plan: ROCm/Megatron-LM is
the required framework (not Microsoft's Megatron-DeepSpeed); RCCL needs
the `aws-ofi-rccl` plugin LD_PRELOAD'd or it defaults to TCP/IP on
Slingshot; `torch.compile + bf16 + ROCm` has known NaN issues on Megatron
layers as of 2025 (FRO §Frontier-specific).

Critical batch size: McCandlish 2018 GNS *underestimates* the true
`B_crit` by orders of magnitude (Marsden 2025); 2507.07101 finds
small-batch training is *more* robust per-FLOP for recurrent models.
Recommendation is 1M–2M tokens global batch with empirical measurement
(FRO §Critical batch size).

---

## 5. M2RNN delta (one paragraph)

M2RNN (Mishra et al., arXiv:2603.14360) validates the broader thesis that
nonlinear matrix-state recurrence is a viable answer to state-capacity
limits in linear RNNs, but its update rule and head geometry differ in two
load-bearing ways. M2RNN computes a candidate `z = tanh(H_{t-1} W + k vᵀ)`,
a forget-interpolated state `H_t = f H_{t-1} + (1 − f) z`, and an output
`y = qᵀ H_t + D v` — that is, the state is a nonlinear hidden state mixed
through a learned transition `W`, with raw outer-product injection and an
additive residual value path (MENU §M2RNN Contrast). Emender/E88 instead uses
a *delta-rule write* `S = tanh(decay·S + k(v − Sᵀk)ᵀ)` — the model reads
from memory, computes an error, writes the correction, and has no `D v`
residual path. M2RNN's published paper-shape is also head-asymmetric: one
shared q/k addressing stream feeds hundreds of value/forget/gate heads,
which (in the M2C local reproduction at 1.27B) collapses gradients through
the narrow q/k path and produces grad norms 10⁶–10⁷; E88's per-head q,k,v
geometry avoids this by construction (M2C §Production 1.27B observations
and §Working hypothesis). The tied/CMA-ES variant of M2RNN that uses many
independent addressing programs *is* stable in the same training setup
(loss 4.085 at step 9250 in M2C), supporting the conclusion that the
instability is the geometry choice, not the matrix-state-RNN family.
For the paper, the defensible narrow claim is therefore: pure Emender can be
trained at production scale, does not need attention or linear-recurrent
layers, and matches or beats the best linear-recurrent baselines on
language-model quality — not the broader "first nonlinear matrix-state
RNN" claim (M2C §Positioning).

---

## 6. Contradictions and open questions

### 6.1 Output gate: on or off

* ABL Round 1 finds `E88a_nogate` slightly improves over baseline
  (Δ=−0.014 avg100); Round 2's `E88b_nonorm` keeps the gate.
* MENU §Current E88 Gate State notes that "later 500M work found SiLU
  output gating clearly better than no gate", and production keeps
  `--use_gate 1`.
* MENU §Architecture Menu / Output Path explicitly flags revalidation at
  1.27B as a "high-value first test."

The code keeps the gate on (`e88_fused.py:149, 173-176, 251-254,
314-316`), but this is the most live unresolved contradiction. The
recommended Phase-1 control (`no output gate` at 1.27B ctx2k) had not
been resolved as of MENU's writing date 2026-05-10.

### 6.2 tanh on the state vs linear state

* ABL Round 1 `E88a_linear` is Δ=−0.004 (essentially tied with the tanh
  baseline at 100M-class). Round 4 `E88d_linear` is also tied with
  `E88c_nogate` ("Linear = Tanh!").
* E63 §The Expressivity-Parallelism Tradeoff insists tanh-on-state is the
  UTM-class hinge — linear-in-h cannot pass Siegelmann–Sontag.
* MENU §State Nonlinearity: "Tanh is the theoretical hinge for nonlinear
  temporal computation, but earlier loss-only runs sometimes showed
  linear state close. Expressivity tasks should decide this, not only
  Pile loss."

Production keeps `tanh` (`e88_fused.py:308`); the contradiction is between
loss-only ablations (which say linear is fine) and expressivity arguments
(which say only tanh gives the computational class). The resolution
proposed by MENU is to evaluate on state-tracking / modular-counter /
parity / FSM-tracking tasks, not only loss.

### 6.3 Convolutions: hurt or help

* ABL Round 1 finds `E88a_noconv` is the winner (Δ=−0.027).
* BAL and MENU do not mention conv at all.
* Production E88 (and `e88_fused.py`) has no short convolution.

Probably consistent — conv-free is the chosen design — but the small
delta suggests the ablation may not have been re-run at production
scale.

### 6.4 Simple vs Mamba2-style decay

* ABL Round 1 finds simple sigmoid decay is +0.031 worse, motivating
  Mamba2-style.
* ABL Round 4 finds `E88d_simpledecay` ties `E88c_nogate` (Δ=−0.002)
  once gate and conv have been removed.
* MENU §Decay says "no decay" should be tested before assuming Mamba2 is
  necessary.

Production keeps Mamba2-style (`e88_fused.py:178-192, 244-248`) and the
STAB doc strongly motivates the log-space formulation for stability, not
just loss. The contradiction is whether Mamba2's *learning dynamics*
benefit (cited in STAB) is what's load-bearing, or just the per-step
*loss*. ABL only measured loss.

### 6.5 ScheduleFree DiLoCo with optimizer-state sync vs model-only

* SFD §Local Smoke Results: at H=50 and H=100, optimizer-state sync gives
  marginally lower drift; by 5K steps the curves tie and model-only is
  faster.
* SFD §ScheduleFree Interaction: "do not average ScheduleFree optimizer
  internals, initially" — because the semantics of merging ScheduleFree
  state are unclear.

No live contradiction, but the open question is whether the tie persists
at 1.27B and at H ≥ 500. The recommended Frontier-pilot setup uses
H=250, model-only.

### 6.6 ParaRNN convergence on matrix state

Untested. FRO §Headline calls this out explicitly: ParaRNN's published
examples are vector-state nonlinear RNNs; for E88's matrix state the
Newton block size is `n²×n² = 1024×1024` per block per layer (n=32),
which is solvable but expensive, and convergence on the
`tanh(decay·S + outer(k, v−Sᵀk))` map is empirically unverified. This
is an open question, not a contradiction.

### 6.7 Production head count vs file defaults

`e88_fused.py:144-153` defaults to `n_heads=104, n_state=32`. The
production 1.27B run uses 370 heads (MENU §E88/Emender Abstract View;
M2C "dim 1664, depth 12, H=370, N=32"). BAL §Recommended Configs lists
the lower-head balanced family (32–64 heads). This is not a
contradiction — file defaults are conservative and production overrides
them via constructor args — but it is worth flagging for anyone reading
`e88_fused.py` in isolation.

---

## 7. Source-doc citations

Each of the 11 source documents is cited at least once above. Index:

| Cite | Document | Used in §§ |
|---|---|---|
| E63 | `docs/E63_NONLINEAR_DELTA_DESIGN.md` | 1, 2 (state nonlinearity), 6.2 |
| ABL | `docs/E88_ABLATION_NOTES.md` | 1, 2, 3, 6.1, 6.2, 6.3, 6.4 |
| BAL | `docs/E88_BALANCED_CONFIG_GUIDE.md` | 1, 2 (head/state geometry), 3 (#5), 6.7 |
| FRO | `docs/FRONTIER_DISTRIBUTED_TRAINING_RESEARCH.md` | 4.3, 4.4, 6.6 |
| M2C | `docs/M2RNN_E88_COMPARISON.md` | 1 (#3 many heads), 3 (#4), 5, 6 |
| TSPEC | `docs/MATRIX_ELMAN_TRITON_SPEC.md` | 4.1 (fusion pattern lineage) |
| MSE | `docs/MATRIX_STATE_ELMAN.md` | 1 (motivation: d² dynamic state for O(d²) cost) |
| MENU | `docs/NDM_ARCHITECTURE_MENU.md` | 1, 2 (output gate), 5, 6.1, 6.2, 6.4, 6.7 |
| SFD | `docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md` | 4.4, 6.5 |
| STAB | `docs/STABILITY_AND_PARAMETERIZATION.md` | 2, 3 (#1, #2) |
| FIX | `docs/STABILITY_FIX_GUIDANCE.md` | 3 (#6: bound the state, not just the output) |

### MSE inline note

MSE is the earliest doc in this dossier. It introduces the
matrix-state-with-outer-product-write idea that E88 inherits — `d²`
dynamic-state capacity for `O(d²)` computational cost via element-wise
ops and outer products instead of `W @ h` matrix-vector multiplies
(MSE §Computational Cost Analysis). The "key as nonlinear address,
value as content" framing of MSE §1 is the precursor to Emender's delta
update, with the addition that Emender also has the explicit `delta` step
(reads what is there, writes the *correction*).

---

*Compiled 2026-05-23 for `synthesize-design-dossier`; see
`docs/MEMORY.md` for the index of related notes.*
