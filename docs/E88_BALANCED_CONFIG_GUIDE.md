# E88 Balanced Configuration Guide

## Benchmark Results (500M, 10 min training)

| Config | Loss | Tok/s | State/Layer | vs Mamba2 |
|--------|------|-------|-------------|-----------|
| E88_b56n32 | **1.55** | 10,766 | 57K | **+0.25** better |
| E88_b60n32 | **1.55** | 10,887 | 61K | **+0.25** better |
| E88_b64n32 | 1.60 | 10,140 | 65K | +0.20 better |
| Mamba2 | 1.80 | 15,557 | 410K | baseline |
| FLA-GDN | 1.29 | 21,137 | 1.33M | best |

**Key insight**: E88 with n_state=32 beats Mamba2 while using 7× less state!

## The Balancing Principle

E88 has a projection bottleneck when `n_heads × n_state >> dim`. The key projections are:

```
qkv_proj: [dim] → [3 × n_heads × n_state]  (Q, K, V projection)
o_proj:   [n_heads × n_state] → [dim]      (output projection)
```

**Rule of thumb**: Keep `n_heads × n_state ≈ dim` (within 1-2×) for balanced performance.

### Projection Ratio

```
projection_ratio = (n_heads × n_state) / dim
```

| Ratio | Status | Example |
|-------|--------|---------|
| 0.5-2.0 | ✅ Balanced | dim=2048, h=64, n=32 → ratio=1.0 |
| 2.0-4.0 | ⚠️ Marginal | dim=512, h=32, n=32 → ratio=2.0 |
| >4.0 | ❌ Bottleneck | dim=384, h=81, n=128 → ratio=27.0 |

### State Size

E88 recurrent state per layer: `n_heads × n_state²`

For comparison:
- Mamba2: ~410K state/layer (128 heads × 3200 dim)
- FLA-GDN: ~1.33M state/layer (4 heads × 576²)

## Balanced Configurations for 500M Parameters

### Calculation Method

1. **Target layer params**: `500M / depth ≈ 15-16M per layer`
2. **E88 layer params** ≈ `4 × dim × n_heads × n_state` (dominant terms)
3. **For balance**: Set `n_heads × n_state = dim`
4. **Solve**: `4 × dim² ≈ 15.6M` → `dim ≈ 2000`

### Recommended Configs (depth=32)

| Config | dim | n_heads | n_state | Ratio | State/Layer | Params |
|--------|-----|---------|---------|-------|-------------|--------|
| E88_b64n32 | 2048 | 64 | 32 | 1.0 | 65,536 | ~530M |
| E88_b48n32 | 1536 | 48 | 32 | 1.0 | 49,152 | ~300M |
| E88_b32n64 | 2048 | 32 | 64 | 1.0 | 131,072 | ~530M |
| E88_b24n64 | 1536 | 24 | 64 | 1.0 | 98,304 | ~300M |
| E88_b48n48 | 2304 | 48 | 48 | 1.0 | 110,592 | ~670M |
| E88_b36n48 | 1728 | 36 | 48 | 1.0 | 82,944 | ~380M |

### Anti-patterns (Avoid These)

| Config | dim | n_heads | n_state | Ratio | Problem |
|--------|-----|---------|---------|-------|---------|
| E88_h81n128 | 384 | 81 | 128 | 27.0 | Extreme expansion, 10x slowdown |
| E88_h100n64 | 640 | 100 | 64 | 10.0 | Heavy expansion |
| E88_h5n128 | 6144 | 5 | 128 | 0.1 | Underutilized state |

## Quick Reference Formula

For a target ~500M params at depth=32:

```python
# Perfect balance: n_heads × n_state = dim
dim = 2048
n_heads = 64
n_state = 32
# Or equivalently:
n_heads = 32
n_state = 64
```

For different param targets:
```python
# dim ≈ sqrt(target_params / (4 × depth))
# Then choose n_heads, n_state such that n_heads × n_state ≈ dim
```
