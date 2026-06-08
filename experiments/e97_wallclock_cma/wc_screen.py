"""e97-wallclock-cma worker: one-config time/token-bounded real-Pile screen with
the gdn2_nonlin_shell head at a chosen chunk-size C, state-nonlinearity and
fused/chunked-reference kernel. Reuses the e97_delta_1p3b_cma screen training loop
(pilot data, schedule-free AdamW, bf16, fast_heldout_bpb) verbatim — only the
model build differs (shell C-axis instead of e97_delta).

cfg keys: dim, head_type_logits, lr, knob_lr_mult, batch_size, bf16,
          shell_state_nonlin, shell_state_chunk, shell_fused, lam_max, beta_max.
REAL DATA / REAL TRAINING. No mocks.
"""
import os, sys, json, math, time, argparse, datetime, gc
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
_DELTA = os.path.join(_ROOT, 'experiments', 'e97_delta_1p3b_cma')
for p in (_ROOT, _DELTA, os.path.join(_ROOT, 'experiments', 'e99_1p3b_cma'), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
import screen as S   # reuse make_optimizer, fast_heldout_bpb, _fast_loss, run loop pieces
from wc_common import build_shell_ladder


def build_model(cfg, device):
    m = build_shell_ladder(cfg['dim'], cfg['head_type_logits'],
                           shell_state_nonlin=cfg.get('shell_state_nonlin', 'tanh'),
                           shell_state_chunk=int(cfg.get('shell_state_chunk', 64)),
                           shell_fused=bool(cfg.get('shell_fused', True)),
                           lam_max=cfg.get('lam_max', 1.585),
                           beta_max=cfg.get('beta_max', 2.747))
    m = m.to(device)
    if cfg.get('bf16', True):
        m = m.bfloat16()
    return m


# Patch screen's build_model so its run()/roundtrip use the shell builder.
S.build_model = build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_json'); ap.add_argument('--out')
    ap.add_argument('--wall_seconds', type=float, default=480.0)
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
    out['shell_state_chunk'] = cfg.get('shell_state_chunk')
    out['shell_state_nonlin'] = cfg.get('shell_state_nonlin')
    out['shell_fused'] = cfg.get('shell_fused')
    json.dump(out, open(args.out, 'w'), indent=2)
    h = out.get('heldout') or {}
    print(f"DONE bpb={h.get('heldout_bpb')} avg_loss={out.get('avg_loss')} "
          f"tokens={out.get('tokens')} tok/s={out.get('sustained_tok_s')} "
          f"C={cfg.get('shell_state_chunk')} nl={cfg.get('shell_state_nonlin')} "
          f"fused={cfg.get('shell_fused')} stop={out.get('stop_reason')}", flush=True)


if __name__ == '__main__':
    main()
