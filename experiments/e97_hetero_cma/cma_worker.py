"""e97-hetero-cma CMA worker: one-config wall-clock real-Pile bpb screen for the
heterogeneous cell. Reuses the e97_delta_1p3b_cma screen.run() training loop
verbatim (schedule-free AdamW, bf16, real Pile, fast_heldout_bpb) but swaps the
model builder for hetero_common.build_hetero_ladder so the CMA can vary the
nonlinear-head FRACTION (e97_delta split-edit per-step-tanh), mlp_ratio, lam/beta
caps, lr and knob_lr_mult. Stream overlap ON; fused split-edit (no eager).

cfg keys: dim, head_type_logits, lr, knob_lr_mult, batch_size, bf16,
          e97_state_nonlin, lam_max, beta_max, mlp_ratio.
REAL DATA / REAL TRAINING. No mocks.
"""
import os, sys, json, argparse, datetime
_THIS = os.path.dirname(os.path.abspath(__file__))
_DELTA = os.path.join(_THIS, '..', 'e97_delta_1p3b_cma')
for p in (_THIS, _DELTA, os.path.join(_THIS, '..', 'e99_1p3b_cma')):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
import screen as S
from hetero_common import build_hetero_ladder, assert_fused_no_eager


def build_model(cfg, device):
    m = build_hetero_ladder(
        cfg['dim'], cfg['head_type_logits'],
        e97_state_nonlin=cfg.get('e97_state_nonlin', 'tanh'),
        use_chunked_e97_delta=bool(cfg.get('use_chunked_e97_delta', False)),
        overlap_streams=bool(cfg.get('overlap_streams', True)),
        lam_max=cfg.get('lam_max', 1.585), beta_max=cfg.get('beta_max', 2.747),
        mlp_ratio=cfg.get('mlp_ratio', None) or (6208 / 2304))
    m = m.to(device)
    if cfg.get('bf16', True):
        m = m.bfloat16()
    # loud guard: the split-edit capability heads must be on the fused Triton path
    from shapes import allocate, BASE
    if allocate(cfg['head_type_logits'], BASE['n_heads'])['counts']['e97_delta'] > 0:
        assert_fused_no_eager(m)
    return m


S.build_model = build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_json'); ap.add_argument('--out')
    ap.add_argument('--wall_seconds', type=float, default=240.0)
    ap.add_argument('--token_cap', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--roundtrip', type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.config_json))
    try:
        out = S.run(cfg, args.wall_seconds, args.token_cap, args.seed, bool(args.roundtrip))
    except torch.cuda.OutOfMemoryError as e:
        out = dict(cfg=cfg, error='OOM', msg=str(e)[:400],
                   timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    json.dump(out, open(args.out, 'w'), indent=2)
    h = out.get('heldout') or {}
    print(f"DONE bpb={h.get('heldout_bpb')} tokens={out.get('tokens')} "
          f"tok/s={out.get('sustained_tok_s')} stop={out.get('stop_reason')} "
          f"err={out.get('error')}", flush=True)


if __name__ == '__main__':
    main()
