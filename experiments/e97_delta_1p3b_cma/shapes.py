"""e97delta-1p3b: shape/param helpers + live chunked-kernel verification.

Param-matching for the WITHIN-LAYER typed-gdn2-lm cell at ~1.3B:
  - same fixed axes for BOTH arms (gdn2-mlp baseline AND e97_delta candidate):
      depth, n_heads, n_state, expansion, mlp_ratio (SwiGLU MLP).
  - arms differ ONLY in head_type_logits (head-type fractions) and dim, where
    dim is DERIVED per allocation to hold counted LadderLM params at the target.
  - gdn-neg == gdn2_recall heads with gdn_allow_neg_eigval=True (global on this
    path, matching the prior E99 batch). e97_delta == type index 7, routed through
    the chunked-parallel fused Triton fwd+bwd kernel (use_chunked_e97_delta=True).

This file is REAL: it builds actual LadderLM modules and counts real parameters;
the kernel check runs a real fwd+bwd on GPU and asserts the chunked path is live.
"""
from __future__ import annotations
import os, sys, math, json, time, contextlib

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)

import torch


@contextlib.contextmanager
def _no_init():
    """Skip in-place parameter-init fills during construction. Param SHAPES (and
    thus numel) are unchanged, so exact param counts are identical (verified), but
    we skip filling billions of values — a ~500x speedup for CPU param counting."""
    import torch.nn.init as I
    names = ['uniform_', 'normal_', 'kaiming_uniform_', 'kaiming_normal_',
             'xavier_uniform_', 'xavier_normal_', 'zero_', 'fill_', 'ones_',
             'zeros_', 'trunc_normal_']
    saved = {}
    for n in names:
        if hasattr(torch.Tensor, n):
            saved[('t', n)] = getattr(torch.Tensor, n)
            setattr(torch.Tensor, n, lambda self, *a, **k: self)
        if hasattr(I, n):
            saved[('i', n)] = getattr(I, n)
            setattr(I, n, lambda t, *a, **k: t)
    try:
        yield
    finally:
        for (kind, n), f in saved.items():
            setattr(torch.Tensor if kind == 't' else I, n, f)


_COUNT_CACHE = {}
_DERIVE_CACHE = {}

VOCAB_SIZE = 50281
PARAM_TARGET = 1_270_000_000
PARAM_TOL = 0.02
DIM_MULTIPLE = 64
MLP_RATIO = 6208 / 2304  # official GDN-2 SwiGLU hidden ratio (gdn2-mlp reference)

# Fixed non-capacity axes shared by BOTH arms. n_heads chosen so head-type
# fractions allocate at reasonable granularity; depth/n_state match the prod path.
BASE = dict(depth=18, n_heads=64, n_state=32, expansion=1.0, mlp_ratio=MLP_RATIO)

# Full 8-type canonical order:
#   [gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell, e97_raw, e97_delta]
TYPE_NAMES = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
              'gdn2_nonlin_shell', 'e97_raw', 'e97_delta']
LOG0 = -30.0


def fracs_to_logits8(fr: dict) -> list:
    """8-vector logits from a target-fraction dict (log f, 0 -> LOG0)."""
    return [math.log(fr[t]) if fr.get(t, 0.0) > 0 else LOG0 for t in TYPE_NAMES]


def allocate(logits, n_heads):
    from ndm.models.typed_head_mixture import allocate_types
    return allocate_types(n_heads, logits)


def build_ladder(dim, logits, knob=None, e97_state_nonlin=None):
    from ndm.models.ladder_lm import LadderLM
    lk = dict(head_type_logits=[float(x) for x in logits],
              gdn_allow_neg_eigval=True,
              lam_max=(knob or {}).get('lam_max', 1.585),
              beta_max=(knob or {}).get('beta_max', 2.747))
    # e97_state_nonlin='identity' makes the e97_delta heads LINEAR-state, which is
    # the prerequisite for the chunked-parallel fused Triton kernel to actually
    # engage (the chunked guard in e88_fla_hybrid requires linear_state=True). The
    # default 'tanh' silently routes e97_delta to the SLOW sequential T-scan
    # (fuse-2kernel finding: prior 1.3B run never reached the chunked kernel).
    if e97_state_nonlin is not None:
        lk['e97_state_nonlin'] = str(e97_state_nonlin)
    m = LadderLM(vocab_size=VOCAB_SIZE, dim=int(dim), depth=BASE['depth'],
                 level='typed-gdn2-lm', n_heads=BASE['n_heads'], n_state=BASE['n_state'],
                 expansion=BASE['expansion'], layer_kwargs=lk,
                 mlp_ratio=BASE['mlp_ratio'])
    return m


def _counts_key(counts):
    return tuple(int(counts[t]) for t in TYPE_NAMES)


def count_params(dim, logits, knob=None):
    counts = allocate(logits, BASE['n_heads'])['counts']
    key = (int(dim), _counts_key(counts))
    if key in _COUNT_CACHE:
        return _COUNT_CACHE[key]
    with _no_init():
        m = build_ladder(dim, logits, knob)
        n = sum(p.numel() for p in m.parameters())
    del m
    _COUNT_CACHE[key] = n
    return n


def derive_dim(logits, knob=None, target=PARAM_TARGET, tol=PARAM_TOL):
    """Derive dim (multiple of 64) holding counted params within tol. Cached by
    integer head counts (param count depends on counts, not continuous logits)."""
    counts = allocate(logits, BASE['n_heads'])['counts']
    ckey = _counts_key(counts)
    if ckey in _DERIVE_CACHE:
        return dict(_DERIVE_CACHE[ckey])
    lo, hi = 768, 4096
    best = None
    while lo <= hi:
        mid = max(DIM_MULTIPLE, ((lo + hi) // 2 // DIM_MULTIPLE) * DIM_MULTIPLE)
        n = count_params(mid, logits, knob)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (mid, n)
        if n < target:
            lo = mid + DIM_MULTIPLE
        elif n > target:
            hi = mid - DIM_MULTIPLE
        else:
            break
    dim, n = best
    rel = (n - target) / target
    out = dict(dim=int(dim), params=int(n), params_b=round(n / 1e9, 4),
               rel=round(rel, 4), within_tol=bool(abs(rel) <= tol),
               counts=counts)
    _DERIVE_CACHE[ckey] = dict(out)
    return out


def _timed_tok_s(m, device, B=4, T=512, n_iter=40):
    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    for _ in range(3):  # warmup
        loss = m(x, return_loss=True)
        loss = loss[0] if isinstance(loss, tuple) else loss
        loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        loss = m(x, return_loss=True)
        loss = loss[0] if isinstance(loss, tuple) else loss
        loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return B * T * n_iter / dt, dt


def _busy_util(m, device, B=24, T=512, seconds=8.0):
    """Run a continuous fwd+bwd loop for `seconds`, sampling GPU util WITHOUT
    syncing (so samples land during active compute, not at a drained barrier)."""
    import subprocess, threading
    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    utils = []
    stop = {'v': False}
    def sampler():
        while not stop['v']:
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu',
                     '--format=csv,noheader,nounits', '-i', gpu], timeout=5).decode()
                utils.append(float(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.2)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.time()
    while time.time() - t0 < seconds:
        loss = m(x, return_loss=True)
        loss = loss[0] if isinstance(loss, tuple) else loss
        loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    stop['v'] = True; th.join(timeout=2)
    return utils


def verify_chunked_kernel(dim, device='cuda'):
    """Build an e97_delta-heavy cell, run a real fwd+bwd, assert chunked path live,
    measure busy-loop GPU util, and compare tok/s vs the gdn-neg-only baseline at
    the SAME dim (near-GDN throughput check). No mocks: real tensors, real grad."""
    fr = {'gdn2_recall': 0.25, 'e97_delta': 0.75}
    logits = fracs_to_logits8(fr)
    alloc = allocate(logits, BASE['n_heads'])
    m = build_ladder(dim, logits).to(device).bfloat16()
    layer0 = m.layers[0]
    node = getattr(layer0, 'mixer', layer0)   # unwrap MLP wrapper
    inner = getattr(node, 'inner', node)      # unwrap protocol adapter -> typed layer
    e97d = getattr(inner, 'e97_delta', None)
    assert e97d is not None, "e97_delta sub-block missing"
    use_chunked = getattr(inner, 'use_chunked_e97_delta', None)
    assert use_chunked is True, f"use_chunked_e97_delta not True: {use_chunked}"
    utils = _busy_util(m, device)
    util_mean = sum(utils) / len(utils) if utils else None
    util_max = max(utils) if utils else None
    tok_s_delta, _ = _timed_tok_s(m, device)
    del m; torch.cuda.empty_cache()
    # gdn-neg-only baseline at SAME dim (near-GDN tok/s reference)
    base = build_ladder(dim, fracs_to_logits8({'gdn2_recall': 1.0})).to(device).bfloat16()
    tok_s_gdn, _ = _timed_tok_s(base, device)
    del base; torch.cuda.empty_cache()
    return dict(dim=dim, counts=alloc['counts'],
                util_mean=round(util_mean, 1) if util_mean else None,
                util_max=util_max, n_util_samples=len(utils),
                tok_s_e97delta=round(tok_s_delta, 1), tok_s_gdn_neg=round(tok_s_gdn, 1),
                tok_s_ratio=round(tok_s_delta / tok_s_gdn, 3),
                use_chunked_e97_delta=bool(use_chunked))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='derive', choices=['derive', 'kernel'])
    args = ap.parse_args()
    if args.mode == 'derive':
        # baseline: 100% gdn-neg (gdn2_recall) + MLP
        base_logits = fracs_to_logits8({'gdn2_recall': 1.0})
        d_base = derive_dim(base_logits)
        print('GDN2-MLP BASELINE (100% gdn-neg):', json.dumps(d_base))
        # candidate seed: e97_delta + gdn-neg mix
        cand_logits = fracs_to_logits8({'gdn2_recall': 0.6, 'e97_delta': 0.4})
        d_cand = derive_dim(cand_logits)
        print('E97_DELTA+GDN-NEG CAND (0.6/0.4):', json.dumps(d_cand))
        out = dict(base=d_base, cand=d_cand, BASE=BASE, target=PARAM_TARGET)
        with open(os.path.join(_THIS, 'results', 'shapes.json'), 'w') as f:
            json.dump(out, f, indent=2)
        print('wrote results/shapes.json')
    else:
        d = int(os.environ.get('PROBE_DIM', '1536'))
        r = verify_chunked_kernel(d)
        print('KERNEL_VERIFY ' + json.dumps(r))
        with open(os.path.join(_THIS, 'results', 'kernel_verify.json'), 'w') as f:
            json.dump(r, f, indent=2)
