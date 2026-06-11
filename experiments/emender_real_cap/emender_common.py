"""emender-real-cap: shared mixture encoding + param-matched dim derivation.

The REAL Emender (this task) = a SEA of gdn2_recall heads (neg-eig on -> recall +
track) plus a SMALL fraction of nonlinear EMENDMENT heads = e97_delta with a
per-step BOUNDED nonlinearity (e97_state_nonlin='tanh', the split-edit
depth-capability head from phi-explore / e97-nonlin-separates-modquad). The
mixture variable searched by CMA is the e97_delta FRACTION f in [0,1]; everything
else (gdn2_recall) fills the remaining heads.

PRECISION (flagged, verified on real GPU): the fused E97 split-edit Triton kernel
is bf16-ONLY -- it refuses fp16 (raises rather than silently running the eager
T-scan) and fp16-autocast silently runs the emendment head in bf16 (a hidden
dtype mismatch). So fp16-uniform-fused is IMPOSSIBLE for the Emender. The
task-sanctioned resolution is bf16 UNIFORM on every arm (half precision, uniform,
fused, no fp32, no mismatch) -- the matched-precision fix to the opt-1p3b fp32
strawman. DTYPE below is the single uniform dtype used on ALL arms.
"""
from __future__ import annotations
import math
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)

# Canonical 9-type order in ndm.models.typed_head_mixture.TYPE_NAMES:
#   0 gdn2_recall 1 e97_track 2 count 3 latch 4 nonlin
#   5 gdn2_nonlin_shell 6 e97_raw 7 e97_delta 8 refit
N_TYPES = 9
IDX_GDN = 0
IDX_E97_DELTA = 7
LOG0 = -30.0  # softmax(-30) ~ 0 -> a type that gets 0 heads

DTYPE = 'bf16'           # uniform half precision on ALL arms (fp16 impossible, see above)
E97_STATE_NONLIN = 'tanh'  # bounded per-step nonlinearity = the split-edit emendment cell


def delta_logits(f_delta: float):
    """9-type logits for the Emender: fraction f_delta of e97_delta, rest gdn2_recall."""
    f = float(min(max(f_delta, 0.0), 1.0))
    L = [LOG0] * N_TYPES
    if f >= 1.0:
        L[IDX_E97_DELTA] = 0.0
        return L
    if f <= 0.0:
        L[IDX_GDN] = 0.0   # pure GDN-2 control
        return L
    L[IDX_GDN] = math.log(1.0 - f)
    L[IDX_E97_DELTA] = math.log(f)
    return L


IDX_E97_TRACK = 1


def composition_logits(a_delta: float, a_track: float):
    """9-type logits for the CMA composition search over the E97 EMENDMENT family
    {e97_delta, e97_track} sprinkled into a SEA of gdn2_recall. gdn2_recall logit is
    pinned at 0 (the reference 'sea'); CMA searches a_delta (e97_delta) and a_track
    (e97_track). softmax over [0, a_delta, a_track] -> fractions; driving both to the
    lower bound recovers pure GDN-2 (the null), keeping them => the real Emender."""
    L = [LOG0] * N_TYPES
    L[IDX_GDN] = 0.0
    L[IDX_E97_DELTA] = float(a_delta)
    L[IDX_E97_TRACK] = float(a_track)
    return L


def head_counts(logits, n_heads):
    from ndm.models.typed_head_mixture import allocate_types
    return allocate_types(n_heads, logits)['counts']


def layer_kwargs_from_logits(logits, lam_max=1.585, beta_max=2.747):
    lk = dict(head_type_logits=[float(x) for x in logits],
              gdn_allow_neg_eigval=True, lam_max=lam_max, beta_max=beta_max,
              e97_state_nonlin=E97_STATE_NONLIN, use_chunked_e97_delta=False,
              overlap_streams=True)
    return lk


def realised_frac(f_delta: float, n_heads: int):
    """Integer-allocated e97_delta fraction for a requested f_delta (largest-remainder)."""
    c = head_counts(delta_logits(f_delta), n_heads)
    return c['e97_delta'] / float(n_heads), c


def layer_kwargs(f_delta: float, lam_max=1.585, beta_max=2.747):
    lk = dict(head_type_logits=delta_logits(f_delta),
              gdn_allow_neg_eigval=True, lam_max=lam_max, beta_max=beta_max)
    if head_counts(delta_logits(f_delta), 64).get('e97_delta', 0) >= 0:
        # the bounded-nonlinearity split-edit emendment head (sequential, overlap-worthy)
        lk['e97_state_nonlin'] = E97_STATE_NONLIN
    return lk


# ---------- param-matched dim derivation (capability HybridLadderLM path) ----------
_CAP_COUNT_CACHE = {}


def cap_count_params(dim, f_delta, depth, n_heads, n_state, vocab,
                     lam_max=1.585, beta_max=2.747):
    key = (int(dim), round(float(f_delta), 6), int(depth), int(n_heads),
           int(n_state), int(vocab))
    if key in _CAP_COUNT_CACHE:
        return _CAP_COUNT_CACHE[key]
    from ndm.models.hybrid_ladder import HybridLadderLM
    lk = dict(head_type_logits=delta_logits(f_delta), gdn_allow_neg_eigval=True,
              lam_max=lam_max, beta_max=beta_max, e97_state_nonlin=E97_STATE_NONLIN)
    m = HybridLadderLM(vocab_size=int(vocab), dim=int(dim), depth=int(depth),
                       layer_pattern=['typed-gdn2'], layer_kwargs=[lk],
                       n_state=int(n_state), n_heads=int(n_heads), expansion=1.0)
    n = sum(p.numel() for p in m.parameters())
    del m
    _CAP_COUNT_CACHE[key] = int(n)
    return int(n)


def derive_cap_dim(f_delta, depth, n_heads, n_state, vocab, target_params,
                   dim_multiple=8, dim_lo=64, dim_hi=2048):
    """Binary-search dim (multiple of dim_multiple) so total params ~= target."""
    lo, hi, best = dim_lo, dim_hi, None
    while lo <= hi:
        mid = max(dim_multiple, ((lo + hi) // 2 // dim_multiple) * dim_multiple)
        p = cap_count_params(mid, f_delta, depth, n_heads, n_state, vocab)
        if best is None or abs(p - target_params) < abs(best[1] - target_params):
            best = (mid, p)
        if p < target_params:
            lo = mid + dim_multiple
        else:
            hi = mid - dim_multiple
    return best  # (dim, actual_params)


# ---------- param-matched dim derivation (LM LadderLM typed-gdn2-lm path) ----------
_LM_COUNT_CACHE = {}


def lm_count_params_logits(dim, logits, depth, n_heads, n_state, vocab,
                           lam_max=1.585, beta_max=2.747, expansion=1.0):
    key = ('lm', int(dim), tuple(round(float(x), 4) for x in logits), int(depth),
           int(n_heads), int(n_state), int(vocab), float(expansion))
    if key in _LM_COUNT_CACHE:
        return _LM_COUNT_CACHE[key]
    from ndm.models.ladder_lm import LadderLM
    lk = layer_kwargs_from_logits(logits, lam_max, beta_max)
    m = LadderLM(vocab_size=int(vocab), dim=int(dim), depth=int(depth),
                 level='typed-gdn2-lm', n_heads=int(n_heads), n_state=int(n_state),
                 expansion=float(expansion), layer_kwargs=lk)
    n = sum(p.numel() for p in m.parameters())
    del m
    _LM_COUNT_CACHE[key] = int(n)
    return int(n)


def lm_count_params(dim, f_delta, depth, n_heads, n_state, vocab,
                    lam_max=1.585, beta_max=2.747):
    return lm_count_params_logits(dim, delta_logits(f_delta), depth, n_heads,
                                  n_state, vocab, lam_max, beta_max)


def derive_lm_dim_logits(logits, depth, n_heads, n_state, vocab, target_params,
                         dim_multiple=64, dim_lo=256, dim_hi=2048, expansion=1.0):
    lo, hi, best = dim_lo, dim_hi, None
    while lo <= hi:
        mid = max(dim_multiple, ((lo + hi) // 2 // dim_multiple) * dim_multiple)
        p = lm_count_params_logits(mid, logits, depth, n_heads, n_state, vocab,
                                   expansion=expansion)
        if best is None or abs(p - target_params) < abs(best[1] - target_params):
            best = (mid, p)
        if p < target_params:
            lo = mid + dim_multiple
        else:
            hi = mid - dim_multiple
    return best


def derive_lm_dim(f_delta, depth, n_heads, n_state, vocab, target_params,
                  dim_multiple=64, dim_lo=256, dim_hi=2048):
    return derive_lm_dim_logits(delta_logits(f_delta), depth, n_heads, n_state,
                                vocab, target_params, dim_multiple, dim_lo, dim_hi)


if __name__ == '__main__':
    # sanity: show counts + param-matched dims across a fraction grid
    nh = 16
    print(f"DTYPE={DTYPE} e97_state_nonlin={E97_STATE_NONLIN} (bf16-uniform, fused)")
    for f in [0.0, 4/64, 8/64, 16/64, 0.5, 1.0]:
        rf, c = realised_frac(f, nh)
        print(f"f_req={f:.4f} -> n_heads={nh} realised_frac={rf:.4f} "
              f"gdn={c['gdn2_recall']} e97_delta={c['e97_delta']}")
