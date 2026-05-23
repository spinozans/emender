# Nonlinear Delta Memory Architecture Menu

Date: 2026-05-10

This note captures the current architectural interpretation of E88 as a
Nonlinear Delta Memory (NDM), contrasts it with M2RNN, and lays out the menu of
ablation paths worth testing before we freeze the next production campaign.

## Working Name

The broad phrase "matrix-state RNN" is no longer specific enough. M2RNN uses a
matrix-valued hidden state, and the phrase is now occupied by nearby work.

The E88-specific identity is better described as:

> a many-head nonlinear delta memory.

Or, as a model family:

> Nonlinear Delta Memory (NDM).

E88 is the current production NDM instance.

## E88 / NDM Abstract View

Each E88 head is a small bounded associative memory. For the current 1.27B run,
each layer has 370 heads, and each head has a 32 x 32 runtime state matrix.

Per token, per head:

```text
retrieved = S^T k
delta     = v - retrieved
S_t       = tanh(decay * S_{t-1} + k delta^T)
y_t       = S_t^T q
y_t       = y_t * silu(g)
```

The core behavior is error-correcting memory:

- `k` addresses the current memory.
- `retrieved` is what the memory already predicts at that address.
- `delta = v - retrieved` is the correction.
- `k delta^T` writes the correction back into state.
- `tanh` bounds the memory itself.
- `q` reads from the updated memory.
- `silu(g)` gates the output contribution, not the write.

This is a plausible abstract copying mechanism. The model can learn to bind
content to an address, retrieve it later, and correct the binding when the
retrieval is wrong. It is not exact transformer-style copy by direct token
attention. It is recurrent copy through a compact, nonlinear, continually
rewritten memory.

## M2RNN Contrast

M2RNN also has matrix state, but the state update has a different meaning:

```text
z_t = tanh(H_{t-1} W + k v^T)
H_t = f_t * H_{t-1} + (1 - f_t) * z_t
y_t = q^T H_t + D v
y_t = y_t * silu(g)
```

The M2 matrix is a nonlinear hidden state transformed by a learned state
transition `W`, injected with an input outer product, then mixed with the old
state by a forget/interpolation gate.

The E88 matrix is a delta-rule memory. It explicitly reads from memory, computes
an error, and writes the correction.

Current M2 paper-shaped design is also head-asymmetric: many value/forget/gate
heads, but usually very few q/k addressing heads. Current E88 is many-headed in
the addressing and memory sense: each head has its own q, k, v, decay, gate, and
state.

## Current E88 Gate State

The production 1.27B E88 run uses:

```text
--use_gate 1
--gate_activation silu
--use_write_gate 0
```

So E88 currently has:

- input-dependent decay: yes
- output gate: yes
- write gate: no
- direct residual `D*v` output path: no

Existing evidence is mixed by scale and era:

- Early small ablations favored no output gate by a small amount.
- Later 500M work found SiLU output gating clearly better than no gate.
- Current optimized kernels and production runs assume SiLU output gating and no
  write gate.

This should be revalidated at 1.27B after the Triton changes.

## Architecture Menu

The interesting knobs are not just "gate or no gate." They define a family of
possible NDM update laws.

### Decay

Current:

```text
S_t = tanh(decay(x_t, h) * S_{t-1} + write)
```

Options:

- no decay: `decay = 1`
- learned constant per-head decay
- input-dependent Mamba2-style decay (current)
- simple sigmoid decay
- coupled write coefficient: `write_scale = 1 - decay`
- decay after write: `S_t = decay * tanh(S_{t-1} + write)`

High-value first test: current input-dependent decay vs no decay vs learned
constant decay. If no decay is close, NDM is even simpler than we think. If it
breaks, decay is a core stability/control mechanism.

### Write Scale / Write Gate

Current:

```text
write = k delta^T
```

Options:

- learned scalar write scale per head, initialized to 1
- learned scalar write scale per layer, initialized to 1
- input-dependent sigmoid beta: `write = beta(x_t) * k delta^T`
- input-dependent signed scale: `write = silu(beta(x_t)) * k delta^T`
- coupled beta: `write = (1 - decay) * k delta^T`

First test should not be full input-dependent write gating. A scalar write scale
is cheaper and tells us whether write amplitude is miscalibrated. Full beta
gating is more expressive but can suppress memory and complicate kernels.

### Output Path

Current:

```text
y = (S^T q) * silu(g)
```

Options:

- no output gate
- SiLU output gate (current)
- sigmoid output gate
- bounded output gate, e.g. `tanh(g)` or `2 * sigmoid(g)`
- direct residual value path: `y = S^T q + D v`
- gated residual value path: `y = S^T q + D v * silu(g_v)`

High-value first test: no output gate at 1.27B ctx2k, same wallclock and same
training recipe. The old evidence is stale.

### State Nonlinearity

Current:

```text
S_t = tanh(pre)
```

Options:

- linear state
- tanh (current)
- softsign
- clipped linear
- leaky tanh / scaled tanh
- separate nonlinearity on write only

Tanh is the theoretical hinge for nonlinear temporal computation, but earlier
loss-only runs sometimes showed linear state close. Expressivity tasks should
decide this, not only Pile loss.

### Delta Rule

Current:

```text
delta = v - S^T k
write = k delta^T
```

Options:

- raw outer product: `write = k v^T`
- normalized delta
- clipped delta
- separate write key and read key
- pre-update read vs post-update read variants
- update from `q` instead of `k`

This is the core identity of NDM. We should be conservative here and test only
small variants until the gate/decay menu is understood.

### Q/K Normalization

Current:

```text
k <- k / ||k||
q <- q / ||q||
```

Options:

- L2 norm (current)
- learned per-head temperature after L2 norm
- RMS norm
- no q norm, keep k norm
- no k norm, keep q norm
- no normalization

Past results say removing L2 normalization is unstable. The useful test is a
learned temperature on normalized q/k, not fully removing norm.

### Head Organization

Current:

- dense update of all heads
- concatenate all head outputs
- project back to model dimension

Options:

- routed readout only
- routed write only
- routed expensive path with cheap passive decay
- top-k memory heads
- grouped heads with shared decay or shared projection
- head load balancing

This is the MoE-like branch. It should come after we settle the basic NDM update
law because routing can hide or amplify instability.

### M2-Like Additions

Possible bridges:

- learned state right-mixing `S W`
- forget interpolation between old state and candidate
- residual `D v` output path
- many value heads with fewer q/k heads

These are useful as controlled comparisons, but they move E88 away from delta
memory. If they help Pile but hurt expressivity, that is an important result.

## Exploration Plan

Use a two-level screen. The first sweep should only use knobs that either
already exist or have a tiny experiment surface:

| Knob | Status |
| --- | --- |
| output gate on/off | implemented |
| linear state | implemented, but needs a non-fused path audit |
| write gate beta | implemented, slower non-optimized path |
| decay mode: `mamba`, `simple`, `none`, `constant` | implemented for E88 probes |
| direct `D v` value residual | implemented for E88 probes |
| scalar write scale | not implemented yet |
| coupled write scale `1 - decay` | not implemented yet |
| q/k temperature | not implemented yet |

### Phase 1: Cheap 1.27B ctx2k Controls

Run short same-recipe probes, not full campaigns:

- baseline current E88: input-dependent decay, SiLU output gate, no write gate
- no output gate
- no decay
- learned constant decay
- scalar write scale
- coupled beta: `write_scale = 1 - decay`
- direct residual `D v`
- linear state

Measure:

- last ~1K-step mean loss
- gradient norm stability
- tok/s
- peak memory
- qualitative slope vs current E88

These should be short enough to fit the free GPUs and identify losers quickly.

### Phase 2: Expressivity Screen

Take the survivors from Phase 1 to the computational expressivity suite:

- state tracking
- selective copy
- parity / modular tasks where available
- long-range algorithmic tasks

Loss parity on Pile is not enough. The architecture claim depends on nonlinear
temporal computation.

### Phase 3: CMA-ES Search

Only after the menu is narrowed:

- rerun E88/NDM CMA-ES with the surviving new knobs
- include context length as a staged axis
- preserve the best current production config as a warm start

### Phase 4: Production Campaign

Redo the 1.27B training campaign only after the update law is stable:

- ctx2k to convergence
- then staged context expansion
- compare E88/NDM, FLA-GDN, Mamba2, and any M2 survivor

## Current GPU Snapshot

As of 2026-05-10 UTC:

- GPU 0: free
- GPU 1: FLA-GDN ctx2k convergence
- GPU 2: Mamba2 ctx2k convergence
- GPU 3: E88 ctx2k convergence
- GPU 4: tied/CMAES-shaped M2RNN ctx2k convergence
- GPU 5: free
- GPU 6: free
- GPU 7: free

So we can run four short E88/NDM controls in parallel without disturbing the
long convergence jobs.

## Immediate Recommendation

Use GPUs 0, 5, 6, and 7 for four short 1.27B ctx2k E88 controls:

1. no output gate
2. no decay
3. scalar write scale, initialized to 1
4. direct residual `D v`

The first two answer the most basic architecture questions. The latter two are
the lowest-risk ways to test whether E88 wants more amplitude control or a
token-local bypass.
