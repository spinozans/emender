"""e97-hetero-cma: shared builders for the heterogeneous E97 cell CMA-ES.

The heterogeneous cell is a within-layer TypedHeadMixtureLayer summing:
  * BULK  : gdn-neg heads (gdn2_recall + gdn_allow_neg_eigval) — fast chunked
            tensor-core scan, carries recall + tracking, ~GDN-2 throughput.
  * NONLINEAR-STATE (depth-capability): e97_delta heads on the SPLIT-EDIT
            recurrence with a PER-STEP bounded saturating state map
            (state_activation='tanh'). phi-explore proved the depth-growing
            capability (modular_quadratic length-extrapolation) needs per-step
            BOUNDED saturation on the SPLIT-EDIT substrate — NOT the gated-delta
            shell (where bounded phi is inert). This head runs the SEQUENTIAL
            split-edit Triton scan (the chunked kernel only engages for linear
            state); hetero-cma extends the side-stream overlap to hide it under
            the chunked bulk so a small nonlinear fraction costs little wall time.

Param-matched at ~1.3B by deriving `dim` per head-count (shapes.derive_dim).
gdn-neg == gdn2_recall (allow_neg_eigval). REAL models, REAL Pile. No mocks.
"""
from __future__ import annotations
import os, sys, time, math

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
_DELTA = os.path.join(_ROOT, 'experiments', 'e97_delta_1p3b_cma')
for p in (_ROOT, _DELTA, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from shapes import (BASE, VOCAB_SIZE, derive_dim, fracs_to_logits8, allocate,
                    TYPE_NAMES)  # noqa: E402


def hetero_fracs(e97_frac: float) -> dict:
    """Target-fraction dict for the bulk/nonlinear split. e97_frac in [0,1]."""
    if e97_frac <= 0:
        return {'gdn2_recall': 1.0}
    return {'gdn2_recall': 1.0 - e97_frac, 'e97_delta': e97_frac}


def build_hetero_ladder(dim, logits, *, e97_state_nonlin='tanh',
                        use_chunked_e97_delta=False, overlap_streams=True,
                        lam_max=1.585, beta_max=2.747, mlp_ratio=BASE['mlp_ratio'],
                        n_heads=None, n_state=None, depth=None):
    """LadderLM (typed-gdn2-lm + SwiGLU MLP) with the blended head mixture.

    Defaults route the e97_delta heads to the SEQUENTIAL split-edit per-step-tanh
    kernel (the depth-capability head) with stream overlap ON. Shape axes default
    to the 1.3B BASE but are overridable for the CMA shape search.
    """
    from ndm.models.ladder_lm import LadderLM
    lk = dict(head_type_logits=[float(x) for x in logits],
              gdn_allow_neg_eigval=True, lam_max=float(lam_max), beta_max=float(beta_max),
              e97_state_nonlin=str(e97_state_nonlin),
              use_chunked_e97_delta=bool(use_chunked_e97_delta),
              overlap_streams=bool(overlap_streams))
    return LadderLM(vocab_size=VOCAB_SIZE, dim=int(dim),
                    depth=int(depth or BASE['depth']), level='typed-gdn2-lm',
                    n_heads=int(n_heads or BASE['n_heads']),
                    n_state=int(n_state or BASE['n_state']),
                    expansion=BASE['expansion'], layer_kwargs=lk,
                    mlp_ratio=float(mlp_ratio))


def timed_tok_s(m, device, B=2, T=2048, n_iter=15, warmup=4):
    """Sustained fwd+bwd tok/s at the real training shape (T=2048)."""
    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    for _ in range(warmup):
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
    return B * T * n_iter / dt


def assert_fused_no_eager(m):
    """Walk the typed layers and assert the e97_delta heads engage the fused
    Triton split-edit kernel (loud guard already lives in _run_e97 during a real
    fwd; here we just confirm the head is present + use_triton)."""
    n_checked = 0
    for layer in m.layers:
        node = getattr(layer, 'mixer', layer)
        inner = getattr(node, 'inner', node)
        e97d = getattr(inner, 'e97_delta', None)
        if e97d is not None:
            assert getattr(e97d, 'use_triton', False), "e97_delta head not on Triton path"
            n_checked += 1
    return n_checked
