# E97 4B llama.cpp CPU port — scope and feasibility (v1)

Date: 2026-08-21 (worker subagent, CPU-only scope task). Authoritative
architecture source: `ndm/models/e88_fla_hybrid.py` (E97 = `E97SplitEditLayer`
config of `E88FLAHybrid`, `use_split_edit=True`), `ndm/models/ladder_lm.py`
(`LadderLM` stack), production kernel `ndm/triton/e88_triton_forward.py`
(SPLIT_EDIT branch). All findings verified against the promoted checkpoint
`pi-native-repair6-arc-segment3-training-v1/checkpoints/checkpoint_agent_sft_u000032_loss_0.5059.pt`
(SHA `d8146498…`, 8,090,809,344 bytes on disk, all 237 tensors bf16).

## 1. Verified op-graph (implementer-ready)

Top level (`LadderLM`, 18 blocks, dim=3840, vocab=50281):

1. `h = x + residual` — residual stream carried in **fp32** (`residual_in_fp32=True`), starts at 0.
2. Pre-norm: `x_n = RMSNorm(h) * ln.weight` — RMSNorm over dim=3840, affine weight only, no bias, eps=1e-5, computed in weight dtype (bf16 at train; fp32 on CPU port).
3. Mixer `mix = E97(x_n)` (below). Block: `mix_out + SwiGLUMLP(RMSNorm(x_n + mix_out) * norm_2.weight)` — note the MLP input norm is over `x_n + mix`, i.e. **norm(norm1(h) + mix)**, NOT llama-style `norm(h + mix)`.
4. `residual = h + mix_out + mlp_out` (fp32); repeat.
5. Final: `logits = lm_head(RMSNorm(x + residual) * norm.weight)`; `lm_head.weight` is **tied** to `embedding.weight` `[50281, 3840]`.

### 1.1 Per-token E97 mixer (production = sequential Triton SPLIT_EDIT path)

Shapes per head: H=60 heads, key/state axis N=64, value axis V=64 (`n_state=64`,
`expansion=1.0`, `key_dim=value_dim=3840`). State `S[h]` is **[64, 64]** per
head — 245,760 fp32 values per layer, 4.42M total (17.7 MB fp32). No
convolutions (`use_conv=False`), no write-gate, no value-residual, no output
norm, no M1/M2 readout arms, `head_mix='concat'`.

Per token t (all inputs bf16 at train; fp32 recommended on CPU):

```
qkv = qkv_proj(x)                       # [3*3840]; q,k = first/second 3840 slices, v = third
q,k,v = silu(q), silu(k), silu(v)       # SiLU applied BEFORE L2 norm (kernel APPLY_SILU_QKV)
# per head h (reshape [H,64]):
k_h = k_h / (||k_h||_2 + 1e-6)          # L2 norm, eps 1e-6 (kernel NORMALIZE_KQ)
q_h = q_h / (||q_h||_2 + 1e-6)
g_logdecay = -exp(A_log[h]) * softplus(a_proj(x)[h] + dt_bias[h])   # fp32 (A_log, dt_bias loaded fp32)
decay_h    = exp(g_logdecay)            # per-head scalar; cast bf16 at train
erase_h    = sigmoid(erase_gate_proj(x) reshaped [H,64])            # key-axis erase/read gate
w_h        = sigmoid(value_write_gate_proj(x) reshaped [H,64])      # value-axis write gate
g_h        = g_proj(x) reshaped [H,64]                              # output gate (SiLU form)

read_key_h   = k_h * erase_h
write_value_h= v_h * w_h
retrieved_h  = sum_n S[h][n,v] * read_key_h[n]          # S^T @ read_key
delta_h      = write_value_h - retrieved_h              # delta correction (raw_write=False)
S[h] = tanh(decay_h * S[h] + outer(delta_h, k_h))       # outer[i,v]=delta_h[v]*k_h[i]; stable tanh (sigmoid form)
out_h = sum_n S[h][n,v] * q_h[n]                        # S^T @ q  (READ uses UPDATED state)
out   = out * silu(g)                                   # fused output gate
mix   = o_proj(concat over h of out_h)                  # [3840,3840]
```

Critical details an implementer must not miss:
- **Read-after-write**: `out` reads the state AFTER the update (same order as kernel).
- SiLU precedes L2 normalization; the outer-product key is the **L2-normalized k** (not the gated read_key).
- The erase gate multiplies only the RETRIEVAL/read key; the write gate multiplies only the write value; tanh receives the gated delta against the un-gated normalized key.
- decay is a per-head **scalar** multiplying the identity part only (delta is NOT decayed; `pos_eigval_clamp=False`).
- tanh must be the numerically stable form (the kernel uses `2*sigmoid(2*pre)-1` because raw exp overflows for |pre|>~44 in fp32).
- Per-head eps additions: `+1e-6` inside L2 norms.

### 1.2 SwiGLU MLP (bias-free, all three weights present)

`mlp(x) = w3( silu(w1(x)) ⊙ w2(x) )` with `w1,w2: [9600,3840]` (gate, up),
`w3: [3840,9600]` (down); hidden 9600 = round(3840*2.5/64)*64.

### 1.3 Tokenizer and vocab

`tokenizer: p50k_base` (tiktoken; vocab exactly **50281** = embedding rows).
Tiktoken cache file `p50k_base.tiktoken` in the tiktoken cache dir; regex
pre-tokenizer differs from GPT-2 r50k (contractions + whitespace classes) —
llama.cpp's GPT-2 BPE path needs a p50k regex override or a custom
pre-tokenizer (see risk R2).

## 2. llama.cpp survey

Verified against current master `src/llama-arch.h` (fetched 2026-08-21):
recurrent/hybrid arch support is mature — `LLM_ARCH_QWEN3NEXT` (Gated
DeltaNet: q/k/v/z gate + beta, Mamba2-style `A_log`/`dt_bias`/`a_proj` decay,
L2-normalized q/k, gated output — **structurally the closest existing arch**;
E97 is a sibling with a tanh state and split-edit gates), `LLM_ARCH_RWKV6/7`,
`LLM_ARCH_NEMOTRON_H`, `LLM_ARCH_GRANITE_HYBRID`, `LLM_ARCH_FALCON_H1`,
`LLM_ARCH_LFM2`, `LLM_ARCH_JAMBA`, `LLM_ARCH_KIMI_LINEAR`. GGML already ships
delta-rule style ops for QWEN3NEXT; E97 needs a new sequential state op
variant (tanh + split-edit). Prefill can run the same op serially over T
(E97's tanh forbids chunk-parallel scan, but CPU decode is T=1 anyway and
prefill of a few K tokens serially in C++ is fine).

Registration surface: add `LLM_ARCH_EMENDER_E97` to llama-arch/llama-model
tensor-name mapping (direct 1:1 with the checkpoint tensor names in §3), a
`build_emender_e97` graph using existing `mul_mat`/`rms_norm`/`silu`/
`sigmoid`/`glue` plus one new op (or a small `llama-building` custom node) for
the 64x64 per-head state update. Inference-only — no backward needed.

## 3. GGUF conversion

Direct tensor mapping (all 237 tensors, no fusion required):
`embedding.weight`→`token_embd` (tied: also `output`), per layer i:
`layers.{i}.norm.weight`→`attn_norm` equivalent, `layers.{i}.mixer.qkv_proj.weight`→fused
qkv `[11520,3840]` (keep fused), `a_proj.weight`→`a` `[60,3840]`,
`A_log`/`dt_bias`→fp32 scalars `[60]` (upcast from bf16), `g_proj.weight`,
`erase_gate_proj.weight`, `value_write_gate_proj.weight`, `o_proj.weight`,
`layers.{i}.norm_2.weight`, `mlp.w1/w2/w3`. Recommended quants: fp32 or q8_0
first (state is tanh-bounded → robust, but must qualify; see §4).

## 4. Numerics requirements (from repo precision authorities)

- `ndm/recurrent_precision.py`: **inference state carry is fp32 even in legacy
  mode** ("Legacy inference has FP32 carry"); production Triton keeps the
  stateful cache in fp32 (`e88_fla_hybrid.py`: "Stateful Triton inference
  keeps the cache in fp32"). CPU port: carry S in fp32, no exceptions.
- `ndm/triton/e88_triton_forward.py`: stable-tanh requirement; 1e-6 L2 eps;
  decay computed fp32 then narrowed (train-only) — on CPU keep decay fp32.
- `docs/validation/e97-bf16-update-diagnostics-v1.md`: bf16 rounding ORDER in
  the recurrence measurably shifts logits (the E88 step kernel was shelved for
  20x-per-layer drift despite fp32 internals). A pure-fp32 CPU recurrence is
  numerically *cleaner* than the production bf16 kernel, but different —
  expect small logit deltas, bounded by the qualification gate, not zero.
- Training-time policy: legacy (bf16 activations, fp32 state carry); the
  promoted checkpoint's `sft_precision` records this. Never run the CPU
  recurrence in bf16.

## 5. Qualification harness (design)

1. **Frozen prompt panel**: 32 prompts — 8 conversation turns, 8 tool-call
   setups (read/bash/edit shape), 8 long-context copy/pointer cases, 8
   adversarial (errors, unicode, code) — stored as JSONL with token ids.
2. **Reference logits**: 1 GPU, brief window (juncture pause is sufficient;
   `scripts/serve_e97_agent_openai.py` exposes logits; or a 20-line
   `load_e97_checkpoint(..., device='cuda')` probe dumping top-64 logits +
   per-token KL vs the checkpoint). Store reference next to the panel.
3. **Acceptance gates** (CPU vs GPU reference, teacher-forced full prompts):
   mean top-1 token agreement ≥ 97% over ≥ 8K generated/forced tokens; mean
   KL(top-64) ≤ 0.05 nats; zero NaN/Inf; greedy continuations judged coherent
   on the 8 conversation cases. Quant ladder: fp32 → q8_0 (same gates) →
   q4_K only if q8_0 passes with margin and a human agrees.
4. **Behavioral spot-check**: replay 3 Stage-B-style tool episodes greedily
   and compare tool-call emission against the GPU server outputs.

## 6. Measured CPU throughput (torch paths, THIS host: 2× AMD EPYC 7713, 256 threads, 1TB RAM)

Promoted checkpoint, fp32 weights, 64 threads, batch 1:

| Path | tok/s | Notes |
|---|---|---|
| `--device cpu` eager (exists today in serve script) | **1.53** | coherent output; Python-dispatch-bound (≈200 op launches/token, ~12 effective GFLOPS vs ~500+ available) |
| `torch.compile` same graph | 0.26 | graph breaks on recurrence loop; worse |
| llama.cpp-style C++ (projected) | **25–60 (fp32)**, **40–80 (q8_0)** | 7.8 GFLOP/token; weights-reading bound: fp32 16GB→~25ms @ ~650GB/s dual-socket; q8_0 ~4.3GB→~7ms; sustained GEMM 300–700 GFLOPS on 64–128 cores |

Per-token FLOP derivation: params/token ≈ 4.0B → ≈7.8 GFLOP/token
(projections+MLP dominated: qkv 88M, gates 3×14.7M, o_proj 14.7M, a_proj
0.46M, MLP 3×36.9M, recurrence state math 2×60×64×64×2=0.98M per layer ×18).
Prefill: same GEMMs are T-parallel; prefill at ~2–5K tok/s (C++).

## 7. Risks

- **R1 (numerics divergence)**: fp32 CPU recurrence vs production bf16-kernel
  fp32-state → small logit deltas; the repo's own history shows rounding-order
  sensitivity. Gate via §5 acceptance thresholds; do not skip.
- **R2 (p50k tokenizer)**: llama.cpp BPE defaults to the GPT-2 regex; p50k
  differs. Wrong pre-tokenization silently shifts behavior. Requires regex
  override in the fork + round-trip verification vs tiktoken over a 10MB
  corpus sample (must be byte-identical ids).
- **R3 (fork maintenance + qualification sprawl)**: a novel arch in a llama.cpp
  fork must be re-based across upstream churn, and every quant/config combo
  wants its own qualification receipt. Mitigate: inference-only fork, single
  recommended quant (q8_0), converter + harness versioned in-repo.
- **R4 (tied embeddings + 50281 odd vocab)**: llama.cpp handles both (GPT-2
  precedent) but padding-free odd vocab needs a correctness check in q8_0.

## 8. Effort estimates and verdict

- **Path A — torch CPU serving (exists today)**: 1.5 tok/s eager. Usable for
  smoke tests now; NOT interactive. A half-day C++/torch extension of just the
  recurrence loop (remove ~200 dispatches/token) projects 8–20 tok/s, but
  stays fp32 (16GB) with no quantization path — moderate ceiling.
- **Path B — llama.cpp port**: GGUF converter 1–1.5d; graph+arch 2–4d;
  tokenizer wiring 0.5–1d; build+qualify 1–2d. **Total ≈ 5–8 focused days**
  to an interactive, quantizable, GPU-free server at 25–80 tok/s.
- **Verdict: GO for Path B as a scoped spike** — converter + fp32 graph +
  §5 harness first, decision gate after the spike reproduces reference logits
  within gates. Path A ships now as the zero-effort fallback (serve script
  `--device cpu`), honestly labeled 1.5 tok/s.
