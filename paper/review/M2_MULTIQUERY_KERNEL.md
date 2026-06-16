# M2 — Higher-Rank Multi-Query Readout (fused kernel) — implementation note

**Task:** `impl-m2-multiquery`. Implements **M2** from
`paper/review/STATE_AWARE_MLP_DESIGN.md` §3: raise the **readout rank** of the
linear-attention matrix state by reading each head's state with **R queries**
instead of one. The genuinely unclaimed axis — no prior linear-attention paper
varies the per-head readout rank via multiple queries.

The **recurrence / state UPDATE is unchanged** (E97 split-edit delta). Only the
**READ** changes: today `out = S^T q` is rank-1 (`V` numbers/head, ~3% of the
`N×V=32×32` state); M2 emits `R` readout vectors/head = an `R`-dim **row**-subspace
of each head's state. `R` is a searchable hyperparameter (1..8). **R=1 reduces
BYTE-IDENTICALLY to the current single-query E97.**

## Why this is cheap (the kernel insight)

In the chunked E97 kernel the query `q` is a **pure readout** — it never enters the
state update `S`, the correction `Delta`, the WY inverse `Tmat`, or the
chunk-to-chunk state propagation. So `R` queries **reuse the entire shared
recurrence** and only re-run the cheap readout matmuls per query:

```
# computed ONCE per chunk (query-independent):
Delta_c, S_entry_c, DA, gamma   # Newton-Schulz Tmat, correction, decay factors
# per query r (cheap):
QK_r  = Q_r @ K^T  ;  A_r = tril(DA * QK_r)
out_r = gamma * (Q_r @ S_entry) + A_r @ Delta_c        # [C, V]
```

This is why measured throughput is **sublinear in R** (below), not the naive `R×`.

## Files

| file | what |
|---|---|
| `ndm/triton/e97_multiquery_autograd.py` | fused fwd (`_e97_fwd_mq_kernel`) + bwd (`_e97_bwd_mq_kernel`) + autograd `E97MultiQueryChunkedFn` + wrapper `e97_multiquery_chunked_triton`. R=1 routes to `e97_delta_chunked_triton`. |
| `tests/test_e97_multiquery.py` | parity vs eager multi-query reference (the oracle) + R=1 byte-identical regression guard. |
| `ndm/models/e88_fla_hybrid.py` | `multiquery_r` ctor arg; `extra_q_proj`; widened `o_proj`; chunked-path M2 branch. |
| `ndm/models/ladder_lm.py` | level `E97-M2` (forces fused chunked split-edit linear-state path). |
| `train.py` | `E97-M2` added to `_e97_family` (auto-Triton + fused-guard). |
| `scripts/cmaes_search_v2.py` | model_type `e97-m2`, `--multiquery_r`, `build_train_command` + `estimate_params_for_config`. |

## Backward VJP (threaded into reverse-replay BPTT)

The backward is the matching VJP, in the SAME reverse-chunk scan as the
single-query kernel. The state gradient `dS` is threaded across chunks unchanged.
The **query-dependent** terms are accumulated over the R queries before the shared
recurrence VJP runs:

- per query, stored: `dQ_r`
- accumulated over r into shared buffers: `dDelta += A_r^T@dOut_r`, the A-side decay
  grad `DD_A += dA_r*QK_r*DA`, `dK += dQK_r^T@Q_r`, the out-term `dS_entry` and
  `dgamma` contributions.
- then computed ONCE: `dTmat → dP/dR/dU`, `dM`, the decay assembly `dg`, and the
  raw-input grads `dk, de, dv, dw`.

## Parity (REAL, on GPU — `tests/test_e97_multiquery.py`, 21/21 pass)

- **R=1 byte-identical** to `e97_delta_chunked_triton` — fwd AND every gradient
  (`torch.equal`) at T ∈ {128, 512, 1024}. Regression guard.
- **R>1 fwd** vs eager multi-query reference: **rel ~2e-6** (fp32) at T ∈
  {128,512,1024}, R ∈ {2,4,8}. bf16 rel < 6e-2 (TF32 tensor-core path).
- **R>1 bwd** vs autograd-through-eager: rel < 5e-3 (fp32), all of k,v,q,decay,e,w.

## Downstream wiring (the chosen "simplest faithful" option)

The kernel emits `out [B,T,H,R,V]`. Wiring (E97 `head_mix='concat'`):

1. **Queries.** Query 0 = the existing (conv+silu+L2-normalized) primary query.
   Queries 1..R-1 come from one fused `extra_q_proj : Linear(dim, (R-1)·key_dim)`,
   each silu/L2-normalized to **match** the primary so the R reads are comparable
   (matches the design-doc M2 "(R-1) extra query projections Linear(dim,key_dim)").
   The short-conv is NOT applied to the extra queries (it is not part of the M2
   spec; the primary query keeps its full processing). To keep the R reads
   comparable, `multiquery_r>1` **asserts `use_conv=False`** (the E88/E97 default) —
   so the primary query also skips conv and all R queries share identical
   processing (one fused `extra_q_proj` + silu + L2).
2. **Gate.** The single output gate `silu(g)` (shape `[B,T,H,V]`) is **broadcast**
   over the R readouts. Simplest; keeps R=1 exact.
3. **Combine = concat → o_proj over R·value_dim.** `out.reshape(B,T,R·value_dim)`
   → `o_proj : Linear(R·value_dim, dim)`. This is the simplest faithful "learned
   combine"; at R=1 it is exactly the baseline `o_proj : Linear(value_dim, dim)`.
   (An optional down-proj before the MLP — the design-doc `m512` variant — is a
   strict generalization left to the iso-param search; the bare concat→o_proj is
   the minimal faithful default.)

## Added params (reported)

Per layer, over the single-query E97 (verified vs real build, < 0.01% est error):

```
M2 extra/layer = (R-1)·dim·key_dim     (extra_q_proj)
               + (R-1)·value_dim·dim    (o_proj widened value_dim→R·value_dim)
               = (R-1)·dim·(key_dim + value_dim)
```

With `key_dim = value_dim = n_heads·n_state`. At the emender-mlp 1.3B geometry
(dim 1792, key=value 6912, depth 11): **+24.77M/layer per extra query ⇒ +272.5M
total for R=2** (before any iso-param rebalance — that is the `cmaes-m2-1p3b`
task's job; `estimate_params_for_config('e97-m2')` already accounts for it).

Small-geometry cross-check (dim 256, H 8, N 32, depth 2): R1 987,936 → R2
1,250,080 → R4 1,774,368 → R8 2,822,944 params (real builds).

## Throughput (REAL, RTX 6000 Ada, fused kernel, emender 1.3B per-layer readout shape B1 T2048 H216 N=V=32 bf16)

| R | fwd vs R1 | fwd+bwd vs R1 |
|---|---|---|
| 1 | 1.00× (0.663 ms fwd / 3.74 ms fwd+bwd) | 1.00× |
| 2 | 1.15× | 0.97× (≈free) |
| 4 | 1.29× | 1.20× |
| 8 | 1.67× | 1.67× |

**Sublinear in R** because the recurrence (Newton-Schulz `Tmat`, `Delta`, state
propagation) is shared; only the readout matmuls repeat — vs the naive
`R`-independent-calls cost of 2×/4×/8×. The full-LM cost adds the dense
`extra_q_proj` + widened `o_proj` (cuBLAS GEMMs, cheap relative to the recurrence).

## Build it (for the CMA driver — `cmaes-m2-1p3b`)

```bash
# arm-based: one rank R per run (design-doc §6 runs distinct arms)
python scripts/cmaes_search_v2.py --model_type e97-m2 --multiquery_r 4 --use_triton_e88 ...
# or directly:
python train.py --level E97-M2 --layer_kwargs '{"multiquery_r": 4}' \
    --n_heads 216 --n_state 32 --expansion 1.0 --use_gate 1 --gate_activation silu \
    --bf16 --use_triton 1 ...
```

The `E97-M2` level self-configures the fused chunked split-edit **linear-state**
path (`use_triton/use_chunked_e97/linear_state` forced). The multi-query readout is
orthogonal to the state nonlinearity; linear-state is the throughput-viable fused
path (per `e97delta-1p3b`/`fuse-2kernel`). The fused-guard asserts no-eager on every
rank for `E97-M2` under bf16.

## Scope / honesty

- Implemented on the **chunked linear-state** E97 kernel (the GDN-2-class throughput
  path). The same readout-loop pattern transfers to the sequential tanh kernel
  (`e88_triton`) if a tanh-state M2 arm is later wanted — the readout VJP there is
  the simpler `dS_t += Σ_r q_r⊗dOut_r`, `dQ_r = S_t@dOut_r`. Not built here.
- N, V ≤ 64; S0 = 0; chunk C=32 (Ada/Ampere SMEM). Same scope as the single-query
  chunked kernel.
- Honest prior (design-doc §7): M2 exposes a richer **linear** readout subspace; a
  capability/BPB win is not guaranteed and must be isolated from capacity by the
  plain-wider-MLP / down-proj controls in `cmaes-m2-1p3b`.
