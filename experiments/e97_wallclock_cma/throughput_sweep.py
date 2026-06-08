"""e97-wallclock-cma Phase B: throughput(C) at the real 1.3B head shape.

Feasibility gate for a wall-clock win: does ANY chunk-size C make the bounded
(tanh) gdn2_nonlin_shell head reach GDN-class throughput? Measures sustained
fwd+bwd tok/s (B=2, T=2048, bf16) for the param-matched dim=2240 LadderLM with
64 heads = 32 gdn-neg + 32 shell, across:
  - pure gdn-neg (baseline, 1 FLA matmul scan)             == speed ceiling
  - shell identity (pure linear shell, 1 FLA call)         == ratio sanity
  - shell tanh fused=True   (single-launch SEQUENTIAL boundary-phi)  ~const in C
  - shell tanh fused=False  (FLA matmul WITHIN C-chunk + tanh boundary), C swept

REAL models / REAL fwd+bwd. No mocks.
"""
import os, sys, json, time, datetime
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _THIS)
import torch
from wc_common import build_shell_ladder, timed_tok_s, fracs_to_logits8

RESULTS = os.path.join(_THIS, 'results')
DIM = 2240


def log(m): print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {m}', flush=True)


def run_one(label, logits, device, **kw):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = build_shell_ladder(DIM, logits, **kw).to(device).bfloat16()
    try:
        tok_s = timed_tok_s(m, device)
        peak = torch.cuda.max_memory_allocated() / 1e6
        r = dict(label=label, tok_s=round(tok_s, 1), peak_mb=round(peak, 1), **{k: v for k, v in kw.items()})
    except torch.cuda.OutOfMemoryError as e:
        r = dict(label=label, tok_s=None, error='OOM', msg=str(e)[:200], **{k: v for k, v in kw.items()})
    del m; torch.cuda.empty_cache()
    log(f'{label}: {r.get("tok_s")} tok/s  peak={r.get("peak_mb")}MB  {kw}')
    return r


def main():
    device = 'cuda'
    gdn = fracs_to_logits8({'gdn2_recall': 1.0})
    mix = fracs_to_logits8({'gdn2_recall': 0.5, 'gdn2_nonlin_shell': 0.5})
    out = []
    # speed ceiling
    out.append(run_one('gdn_neg_pure', gdn, device))
    # pure-linear shell (C=inf endpoint): identity routes to 1 FLA call
    out.append(run_one('shell_identity', mix, device, shell_state_nonlin='identity'))
    # sequential fused tanh: throughput ~const across C (sanity at C=64)
    out.append(run_one('shell_tanh_fused_C64', mix, device,
                        shell_state_nonlin='tanh', shell_state_chunk=64, shell_fused=True))
    # chunked-reference tanh: matmul within C-chunk, phi at boundary. Sweep C.
    for C in (16, 32, 64, 128, 256, 512):
        out.append(run_one(f'shell_tanh_chunkref_C{C}', mix, device,
                           shell_state_nonlin='tanh', shell_state_chunk=C, shell_fused=False))
    ceiling = next((r['tok_s'] for r in out if r['label'] == 'gdn_neg_pure'), None)
    for r in out:
        r['ratio_vs_gdn'] = round(r['tok_s'] / ceiling, 3) if (r.get('tok_s') and ceiling) else None
    res = dict(task='e97-wallclock-cma', phase='throughput(C)', dim=DIM,
               heads='32 gdn-neg + 32 shell', B=2, T=2048,
               gdn_ceiling_tok_s=ceiling, results=out,
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    json.dump(res, open(os.path.join(RESULTS, 'throughput_sweep.json'), 'w'), indent=2)
    log('=== THROUGHPUT(C) vs GDN ceiling ===')
    for r in out:
        log(f'  {r["label"]:28s} {str(r.get("tok_s")):>9} tok/s  ratio={r.get("ratio_vs_gdn")}')
    log(f'WROTE {RESULTS}/throughput_sweep.json')


if __name__ == '__main__':
    main()
