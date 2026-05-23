# E88 Ablation Study Notes

## Summary

Starting from E88sh16n32 (16 heads, 32x32 square state, expansion=1.0), systematically removing components to find the minimal effective architecture.

## Round 1: Basic Component Ablation
**Baseline:** E88sh16n32 (avg100=1.795)

| Variant | Avg100 | Δ | Notes |
|---------|--------|---|-------|
| E88a_noconv | 1.769 | -0.027 | **Winner** - no short convolutions |
| E88a_nogate | 1.782 | -0.014 | No output gating |
| E88a_noconv_nogate | 1.788 | -0.007 | Combined |
| E88a_linear | 1.791 | -0.004 | Linear state (no tanh) |
| E88sh16n32 | 1.795 | baseline | Full E88 |
| E88a_minimal | 1.821 | +0.025 | No conv/gate + simple decay |
| E88a_simpledecay | 1.826 | +0.031 | Simple sigmoid decay |

**Findings:**
- Convolutions hurt slightly (removing helps)
- Gate doesn't help much
- Tanh vs linear is basically a wash
- Mamba2-style decay is important (simple decay hurts)

## Round 2: Deeper Ablation from E88a_noconv
**Baseline:** E88a_noconv (avg100=1.788)

| Variant | Avg100 | Δ | Notes |
|---------|--------|---|-------|
| E88b_nonorm | 1.688 | -0.100 | **Winner** - no output RMSNorm |
| E88b_nogate_nonorm | 1.696 | -0.092 | No gate + no norm |
| E88a_noconv | 1.788 | baseline | |
| E88b_nosilu | 2.094 | +0.307 | No SiLU (much worse) |
| E88b_nol2 | NaN | - | No L2 norm (unstable!) |
| E88b_nosilu_nol2 | NaN | - | Unstable |

**Findings:**
- **Output RMSNorm hurts!** Big improvement from removing it
- SiLU is critical for stability/performance
- L2 normalization is critical (NaN without it)

## Round 3: Head/State Configuration
**Baseline:** E88b_nonorm (avg100=1.706)

| Variant | Avg100 | Δ | Notes |
|---------|--------|---|-------|
| E88c_nogate | 1.695 | -0.011 | **Winner** - no gate |
| E88b_nonorm | 1.706 | baseline | |
| E88c_simpledecay | 1.710 | +0.004 | Simple decay (now ~tied) |
| E88c_h8n64 | 1.819 | +0.113 | 8 heads x 64 (worse) |
| E88c_h32n16 | NaN | - | 32 heads x 16 (unstable) |
| E88c_h24n24 | NaN | - | 24 heads x 24 (unstable) |

**Findings:**
- Removing gate helps slightly
- Simple decay nearly tied when norm removed
- More heads with smaller state causes instability (n_state < 32)
- Fewer heads with larger state is worse

## Current Best Config: E88c_nogate
```python
E88FLAHybrid(
    n_heads=16,
    n_state=32,
    expansion=1.0,      # square state
    use_conv=False,     # no short convolutions
    use_gate=False,     # no output gating
    use_output_norm=False,  # no RMSNorm
    # Keep defaults:
    use_silu=True,      # critical
    use_l2_norm=True,   # critical
    simple_decay=False, # Mamba2-style decay
    linear_state=False, # keep tanh for expressivity
)
```

**Total improvement:** ~0.10 nats from original E88sh16n32

## Components That Matter
- ✅ **Tanh** - computational expressivity (UTM capability)
- ✅ **SiLU** - stability and performance
- ✅ **L2 norm on k/q** - critical for stability
- ✅ **Mamba2-style decay** - slightly better than simple sigmoid
- ❌ **Convolutions** - slight improvement without them
- ❌ **Output gating** - slight improvement without it
- ❌ **Output RMSNorm** - big improvement without it!

## Round 4: Parameter Efficiency
**Baseline:** E88c_nogate (avg100=1.709, 74M params)

| Variant | Avg100 | Params | Δ | Notes |
|---------|--------|--------|---|-------|
| E88d_h12 | 1.632 | 48M | -0.077 | **Winner!** 12 heads |
| E88d_simpledecay | 1.707 | 74M | -0.002 | Simple decay ~tied |
| E88c_nogate | 1.709 | 74M | baseline | |
| E88d_linear | 1.709 | 74M | 0.000 | Linear = Tanh! |
| E88d_tiekv | 1.712 | 56M | +0.003 | tie_kv saves 18M params |
| E88d_h20 | 1.754 | 100M | +0.045 | More heads hurts |

**Major findings:**
1. **12 heads beats 16 heads** - despite fewer params!
2. **Linear state = Tanh state** - no difference at all
3. **tie_kv is efficient** - minimal loss for 24% param reduction
4. **Simple decay now competitive** when other components removed

**Open question:** Is h12 winning because fewer params = less overfitting, or is 12 heads genuinely better? Need equal-param comparison.

## Round 5 (In Progress)
Testing equal-param configurations...
