#!/usr/bin/env python3
"""Aggregate E99 LM-CMA .done evals into machine-readable candidate records.

Reads the harness output dir (eval_*/{.done,params.json,stdout.txt}) and emits:
  - candidates.json  : full per-candidate records
  - candidates.csv   : flat table EXTENDING the handoff summary columns
    (dim,n_heads,n_state,depth,lr,batch_size | AvgLoss | Final | + LM-path extras)

Fields use the handoff names (dim,n_heads,n_state,depth,lr,batch_size). AvgLoss =
mean over the train window (CMA fitness, handoff convention); Final = last-100 avg.
Throughput/walltime recorded so wallclock-matched <-> token-matched can be
cross-walked. Real data only — every number comes from a real .done/stdout.
"""
import os, sys, json, re, glob, csv, argparse, statistics


def parse_stdout_metrics(eval_dir):
    """Median tok/s (post-warmup), step count, last elapsed_h, last raw loss."""
    out = os.path.join(eval_dir, 'stdout.txt')
    toks, n_lines, max_step, last_h, last_loss = [], 0, 0, None, None
    if not os.path.exists(out):
        return dict(median_tok_s=None, n_steps=0, train_minutes=None, last_step_loss=None)
    with open(out) as f:
        for line in f:
            if line.startswith('step'):
                n_lines += 1
                ms = re.match(r'step\s+([0-9]+)', line)
                if ms:
                    max_step = max(max_step, int(ms.group(1)))
                mt = re.search(r'tok/s\s+([0-9.]+)', line)
                if mt and n_lines > 3:
                    toks.append(float(mt.group(1)))
                ml = re.search(r'loss\s+([0-9.]+)', line)
                if ml:
                    last_loss = float(ml.group(1))
                mh = re.search(r'elapsed_h\s+([0-9.]+)', line)
                if mh:
                    last_h = float(mh.group(1))
    return dict(
        median_tok_s=(round(statistics.median(toks), 1) if toks else None),
        n_steps=max_step,                       # real steps (last logged step number)
        last_step_loss=last_loss,               # instantaneous end-of-window loss
        train_minutes=(round(last_h * 60.0, 2) if last_h is not None else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--full_run_tokens', type=float, default=100e9,
                    help='Token budget for projected full-run GPU-day cost')
    args = ap.parse_args()

    rows = []
    for done in sorted(glob.glob(os.path.join(args.output, 'eval_*', '.done'))):
        eval_dir = os.path.dirname(done)
        m = re.search(r'eval_(\d+)', eval_dir)
        eval_id = int(m.group(1)) if m else None
        with open(done) as f:
            d = json.load(f)
        p = d.get('params', {})
        met = parse_stdout_metrics(eval_dir)
        bs = d.get('batch_size') or p.get('batch_size')
        tok_s = met['median_tok_s']
        tokens = (met['n_steps'] * (bs or 0) * (args.chunk_size + 1)) if met['n_steps'] else None
        proj_gpu_days = (args.full_run_tokens / tok_s / 86400) if tok_s else None
        rows.append(dict(
            eval_id=eval_id,
            dim=p.get('dim'), n_heads=p.get('n_heads'), n_state=p.get('n_state'),
            depth=p.get('depth'), lr=p.get('lr'), batch_size=bs,
            actual_params=d.get('actual_params'),
            params_b=(round(d['actual_params'] / 1e9, 4) if d.get('actual_params') else None),
            avg_loss=d.get('loss'),                 # handoff: Best avg loss
            final_loss=d.get('final_loss'),         # handoff: Best final loss
            last_step_loss=met.get('last_step_loss'),  # instantaneous end-of-window loss
            median_tok_s=tok_s, n_steps=met['n_steps'], tokens=tokens,
            train_minutes=met['train_minutes'],
            proj_full_run_gpu_days=(round(proj_gpu_days, 3) if proj_gpu_days else None),
            success=d.get('success'), error=d.get('error'),
        ))

    rows.sort(key=lambda r: (r['avg_loss'] is None, r['avg_loss'] if r['avg_loss'] is not None else 9e9))

    with open(os.path.join(args.output, 'candidates.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)

    cols = ['eval_id', 'dim', 'n_heads', 'n_state', 'depth', 'lr', 'batch_size',
            'params_b', 'avg_loss', 'final_loss', 'last_step_loss', 'median_tok_s',
            'n_steps', 'tokens', 'train_minutes', 'proj_full_run_gpu_days', 'success', 'error']
    with open(os.path.join(args.output, 'candidates.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    ok = [r for r in rows if r.get('success') and r['avg_loss'] is not None and r['avg_loss'] < 10.0]
    print(f"aggregated {len(rows)} evals ({len(ok)} successful) -> candidates.json/.csv")
    for r in rows[:5]:
        print(f"  eval {r['eval_id']}: AvgLoss={r['avg_loss']} Final={r['final_loss']} "
              f"| dim={r['dim']} nh={r['n_heads']} ns={r['n_state']} depth={r['depth']} "
              f"lr={r['lr']} | {r['params_b']}B | {r['median_tok_s']}tok/s")
    return rows


if __name__ == '__main__':
    main()
