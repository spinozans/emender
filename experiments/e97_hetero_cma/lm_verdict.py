"""e97-lm-1p3b: FINAL LM VERDICT for the CMAES-best heterogeneous E97 cell at 1.3B.

Trains the best heterogeneous config from e97-hetero-cma
  H = 48 gdn-neg (gdn2_recall, allow_neg_eigval) + 16 e97_delta (split-edit, per-step
      tanh state) — the depth-capability cell — at dim=2176 (~1.247B, -1.8%),
against the matched baselines
  G = gdn2-mlp (64 gdn-neg + SwiGLU MLP) at dim=2240 (~1.259B, -0.9%)
  L = LSTM reference (gated additive cell) param-matched to ~1.27B,
on held-out BPB, REAL Pile, fused split-edit Triton kernel (no eager, loud guard),
under BOTH matching protocols, 2 seeds (LSTM 1 seed reference):

  * WALL-CLOCK-matched : every arm trains for the SAME wall_seconds. The slower H
    (0.731x GDN-2, e97-hetero-cma) consumes fewer tokens. Final held-out BPB.
  * TOKEN-matched       : every arm trains to the SAME token_cap N_H (= the tokens
    H reaches in the wall arm). Isolates sample-efficiency; H pays the wall penalty.

Reuses the e97_delta_1p3b_cma screen.run() training loop VERBATIM (schedule-free
AdamW, bf16, real Pile, fast_heldout_bpb at the chunked/sequential-kernel speed) and
only swaps the model builder per arch. REAL DATA / REAL TRAINING. No mocks.
"""
import os, sys, json, argparse, datetime, math

_THIS = os.path.dirname(os.path.abspath(__file__))
_DELTA = os.path.join(_THIS, '..', 'e97_delta_1p3b_cma')
for p in (_THIS, _DELTA):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import screen as S
from shapes import (derive_dim, fracs_to_logits8, BASE, VOCAB_SIZE, _no_init,
                    allocate)
from hetero_common import hetero_fracs, build_hetero_ladder, assert_fused_no_eager

# ---- CMA-best optimizer / gate knobs (e97-hetero-cma §6), shared by H and G so the
#      two arms differ ONLY in head mixture + the param-matched dim. ----
CMA_KNOBS = dict(lr=8e-4, knob_lr_mult=11.0, lam_max=1.30, beta_max=3.35)
E97_FRAC = 16 / 64  # the capability knee


def lstm_dim(target=1_270_000_000, tol=0.02):
    """Param-match an LSTM-level LadderLM to ~1.27B on a multiple-of-8 dim grid."""
    from ndm.models.ladder_lm import LadderLM

    def count(dim):
        with _no_init():
            m = LadderLM(vocab_size=VOCAB_SIZE, dim=int(dim), depth=BASE['depth'],
                         level='lstm', mlp_ratio=BASE['mlp_ratio'])
            n = sum(p.numel() for p in m.parameters())
        del m
        return n
    best = None
    for dim in range(1024, 3072, 8):
        n = count(dim)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (dim, n)
        if n > target and best[0] < dim:
            break
    dim, n = best
    rel = (n - target) / target
    return dict(dim=int(dim), params=int(n), params_b=round(n / 1e9, 4),
                rel=round(rel, 4), within_tol=bool(abs(rel) <= tol))


def make_cfg(arch):
    """Return a screen cfg dict + param report for arch in {H, G, L}."""
    if arch == 'H':
        logits = fracs_to_logits8(hetero_fracs(E97_FRAC))
        d = derive_dim(logits)
        cfg = dict(arch='H', dim=d['dim'], head_type_logits=logits,
                   e97_state_nonlin='tanh', use_chunked_e97_delta=False,
                   overlap_streams=True, mlp_ratio=BASE['mlp_ratio'],
                   batch_size=2, bf16=True, **CMA_KNOBS)
        return cfg, d
    if arch == 'G':
        logits = fracs_to_logits8(hetero_fracs(0.0))
        d = derive_dim(logits)
        cfg = dict(arch='G', dim=d['dim'], head_type_logits=logits,
                   e97_state_nonlin='tanh', use_chunked_e97_delta=False,
                   overlap_streams=True, mlp_ratio=BASE['mlp_ratio'],
                   batch_size=2, bf16=True, **CMA_KNOBS)
        return cfg, d
    if arch == 'L':
        d = lstm_dim()
        cfg = dict(arch='L', dim=d['dim'], level='lstm',
                   head_type_logits=fracs_to_logits8(hetero_fracs(0.0)),  # unused
                   mlp_ratio=BASE['mlp_ratio'], batch_size=2, bf16=True,
                   lr=CMA_KNOBS['lr'], knob_lr_mult=1.0)
        return cfg, d
    raise ValueError(arch)


def _build_model(cfg, device):
    """Dispatch builder by arch; install as screen.build_model so run() uses it."""
    arch = cfg.get('arch', 'H')
    if arch == 'L':
        from ndm.models.ladder_lm import LadderLM
        m = LadderLM(vocab_size=VOCAB_SIZE, dim=int(cfg['dim']), depth=BASE['depth'],
                     level='lstm', mlp_ratio=BASE['mlp_ratio'])
        m = m.to(device)
        if cfg.get('bf16', True):
            m = m.bfloat16()
        return m
    # H / G : the typed-gdn2-lm hetero builder (e97_frac=0 -> pure gdn2-mlp)
    m = build_hetero_ladder(
        cfg['dim'], cfg['head_type_logits'],
        e97_state_nonlin=cfg.get('e97_state_nonlin', 'tanh'),
        use_chunked_e97_delta=bool(cfg.get('use_chunked_e97_delta', False)),
        overlap_streams=bool(cfg.get('overlap_streams', True)),
        lam_max=cfg.get('lam_max', 1.585), beta_max=cfg.get('beta_max', 2.747),
        mlp_ratio=cfg.get('mlp_ratio'))
    m = m.to(device)
    if cfg.get('bf16', True):
        m = m.bfloat16()
    # loud guard: split-edit capability heads MUST be on the fused Triton path
    if allocate(cfg['head_type_logits'], BASE['n_heads'])['counts']['e97_delta'] > 0:
        assert_fused_no_eager(m)
    return m


S.build_model = _build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=['H', 'G', 'L'])
    ap.add_argument('--protocol', required=True, choices=['wall', 'token'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--wall_seconds', type=float, default=720.0)
    ap.add_argument('--token_cap', type=int, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg, dreport = make_cfg(args.arch)
    # wall arm: pure wall cap. token arm: token cap with a generous wall ceiling so
    # the slower arms still reach N (1.6x wall headroom over the matched-token est).
    if args.protocol == 'wall':
        wall_s, tok_cap = args.wall_seconds, None
    else:
        wall_s, tok_cap = args.wall_seconds * 2.0, args.token_cap

    try:
        out = S.run(cfg, wall_s, tok_cap, args.seed, do_roundtrip=False)
    except torch.cuda.OutOfMemoryError as e:
        out = dict(cfg=cfg, error='OOM', msg=str(e)[:400])
    out['arch'] = args.arch
    out['protocol'] = args.protocol
    out['param_report'] = dreport
    out['timestamp'] = datetime.datetime.utcnow().isoformat() + 'Z'
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2)
    h = out.get('heldout') or {}
    print(f"DONE arch={args.arch} proto={args.protocol} seed={args.seed} "
          f"bpb={h.get('heldout_bpb')} tokens={out.get('tokens')} "
          f"tok/s={out.get('sustained_tok_s')} wall_min={out.get('wall_minutes')} "
          f"params_b={out.get('params_b')} stop={out.get('stop_reason')} "
          f"err={out.get('error')}", flush=True)


if __name__ == '__main__':
    main()
