"""e97-hetero-cma: one-config blended throughput measurement (clean process).

Builds the 1.3B heterogeneous cell at a given e97_delta (split-edit per-step-tanh)
fraction and overlap setting, measures sustained fwd+bwd tok/s at T=2048 bf16.
Separate process per config so each build gets a clean allocator. REAL model.
"""
import os, sys, json, argparse
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS); sys.path.insert(0, os.path.join(_THIS, '..', 'e97_delta_1p3b_cma'))
import torch
from hetero_common import (build_hetero_ladder, hetero_fracs, timed_tok_s,
                           assert_fused_no_eager)
from shapes import fracs_to_logits8, derive_dim, allocate, BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e97_frac', type=float, required=True)
    ap.add_argument('--overlap', type=int, default=1)
    ap.add_argument('--state_nonlin', default='tanh')
    ap.add_argument('--head_kind', default='split', choices=['split', 'shell'],
                    help='split=e97_delta split-edit (capability head); '
                         'shell=gdn2_nonlin_shell gated-delta (hetero-kernel head)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=2)
    ap.add_argument('--T', type=int, default=2048)
    ap.add_argument('--n_iter', type=int, default=20)
    args = ap.parse_args()
    if args.head_kind == 'shell':
        f = args.e97_frac
        fr = {'gdn2_recall': 1.0} if f <= 0 else {'gdn2_recall': 1.0 - f, 'gdn2_nonlin_shell': f}
    else:
        fr = hetero_fracs(args.e97_frac)
    lg = fracs_to_logits8(fr)
    der = derive_dim(lg)
    dim = der['dim']
    counts = allocate(lg, BASE['n_heads'])['counts']
    chunked = (args.state_nonlin == 'identity')
    if args.head_kind == 'shell':
        # gated-delta shell head (hetero-kernel 0.954x head), per-step bounded (C=1)
        sys.path.insert(0, os.path.join(_THIS, '..', 'e97_wallclock_cma'))
        from wc_common import build_shell_ladder
        m = build_shell_ladder(dim, lg, shell_state_nonlin=args.state_nonlin,
                               shell_state_chunk=1, shell_fused=True).cuda().bfloat16()
    else:
        m = build_hetero_ladder(dim, lg, e97_state_nonlin=args.state_nonlin,
                                use_chunked_e97_delta=chunked,
                                overlap_streams=bool(args.overlap)).cuda().bfloat16()
        if counts['e97_delta'] > 0:
            assert_fused_no_eager(m)
    m.train()
    try:
        ts = timed_tok_s(m, 'cuda', B=args.B, T=args.T, n_iter=args.n_iter)
        peak = torch.cuda.max_memory_allocated() / 1e6
        out = dict(e97_frac=args.e97_frac, overlap=bool(args.overlap),
                   state_nonlin=args.state_nonlin, dim=dim, counts=counts,
                   params_b=der['params_b'], tok_s=round(ts, 1),
                   peak_mb=round(peak, 1))
    except torch.cuda.OutOfMemoryError as e:
        out = dict(e97_frac=args.e97_frac, overlap=bool(args.overlap), error='OOM',
                   msg=str(e)[:200])
    json.dump(out, open(args.out, 'w'), indent=2)
    print('TPUT ' + json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
