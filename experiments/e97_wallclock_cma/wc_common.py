"""e97-wallclock-cma: shared builders for the chunk-size (C) free-axis study.

Reuses the param-matched 1.3B within-layer harness from e97_delta_1p3b_cma
(shapes/BASE/derive_dim/pilot) but routes the gdn2_nonlin_shell head with the
chunk-size C, state-nonlinearity and fused/chunked-reference knobs so C can be
swept as a FREE axis between C=1 (per-step bounding, max edge / min speed) and
C->T (pure linear, no edge / max matmul throughput).

REAL models, REAL Pile. No mocks.
"""
from __future__ import annotations
import os, sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
_DELTA = os.path.join(_ROOT, 'experiments', 'e97_delta_1p3b_cma')
for p in (_ROOT, _DELTA, os.path.join(_ROOT, 'experiments', 'e99_1p3b_cma')):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from shapes import BASE, VOCAB_SIZE, derive_dim, allocate, fracs_to_logits8  # noqa: E402


def build_shell_ladder(dim, logits, *, shell_state_nonlin='tanh',
                       shell_state_chunk=64, shell_fused=True,
                       lam_max=1.585, beta_max=2.747):
    """LadderLM (typed-gdn2-lm + SwiGLU MLP) with the gdn2_nonlin_shell head
    configured for the chunk-size study. gdn-neg == gdn2_recall (allow_neg_eigval).
    """
    from ndm.models.ladder_lm import LadderLM
    lk = dict(head_type_logits=[float(x) for x in logits],
              gdn_allow_neg_eigval=True, lam_max=lam_max, beta_max=beta_max,
              shell_state_nonlin=str(shell_state_nonlin),
              shell_state_chunk=int(shell_state_chunk),
              shell_fused=bool(shell_fused))
    return LadderLM(vocab_size=VOCAB_SIZE, dim=int(dim), depth=BASE['depth'],
                    level='typed-gdn2-lm', n_heads=BASE['n_heads'],
                    n_state=BASE['n_state'], expansion=BASE['expansion'],
                    layer_kwargs=lk, mlp_ratio=BASE['mlp_ratio'])


def timed_tok_s(m, device, B=2, T=2048, n_iter=15, warmup=4):
    """Sustained fwd+bwd tok/s at the real training shape (T=2048)."""
    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    import time
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
