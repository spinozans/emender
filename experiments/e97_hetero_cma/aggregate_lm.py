"""e97-lm-1p3b: aggregate the LM-verdict runs into the BPB tables + verdict.

Reads results/lm_verdict/*.json (one per arch x protocol x seed) and emits a
markdown summary + results/lm_verdict/summary.json. H token-matched reuses H's
wall run (the wall run stops at exactly N_H tokens = the matched-token budget).
REAL numbers only; missing/errored runs are reported as such, never invented.
"""
import os, json, glob, math

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lm_verdict')
NAME = {'H': 'hetero (48 gdn-neg + 16 e97_delta split-edit-tanh)',
        'G': 'gdn2-mlp (64 gdn-neg + SwiGLU MLP)',
        'L': 'LSTM reference'}


def load(tag):
    p = os.path.join(OUT, tag + '.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def bpb(d):
    if not d:
        return None
    h = d.get('heldout') or {}
    return h.get('heldout_bpb')


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 5) if xs else None


def fmt(x, n='—'):
    return n if x is None else x


def main():
    runs = {t: load(t) for t in (
        'H_wall_s0', 'H_wall_s1', 'G_wall_s0', 'G_wall_s1', 'L_wall_s0',
        'G_token_s0', 'G_token_s1', 'L_token_s0')}
    # H token-matched == H wall (wall run stops at N_H tokens)
    runs['H_token_s0'] = runs['H_wall_s0']
    runs['H_token_s1'] = runs['H_wall_s1']

    lines = []
    P = lines.append

    # --- param table ---
    P('### Exact parameter counts (target 1.27B, +/-2%)\n')
    P('| arch | dim | params | rel to 1.27B | within +/-2% | head counts |')
    P('|---|---:|---:|---:|:---:|---|')
    for a in ('H', 'G', 'L'):
        d = runs.get(f'{a}_wall_s0')
        if not d:
            P(f'| {a} | — | MISSING | | | |'); continue
        pr = d.get('param_report', {})
        cnt = d.get('counts', {})
        cs = ', '.join(f'{k}:{v}' for k, v in cnt.items() if v) if cnt else 'lstm'
        P(f"| {NAME[a].split(' (')[0]} | {pr.get('dim')} | {d.get('params_b')}B | "
          f"{pr.get('rel')} | {'YES' if pr.get('within_tol') else 'NO'} | {cs} |")

    # --- throughput ---
    P('\n### Throughput (sustained LM-loop fwd+bwd tok/s, real Pile, B=2 T=2048)\n')
    P('| arch | seed | tok/s | vs gdn2-mlp |')
    P('|---|---:|---:|---:|')
    g_ts = mean([runs[f'G_wall_s{s}']['sustained_tok_s'] for s in (0, 1)
                 if runs.get(f'G_wall_s{s}')])
    for a in ('G', 'H', 'L'):
        for s in (0, 1):
            d = runs.get(f'{a}_wall_s{s}')
            if not d:
                continue
            ts = d.get('sustained_tok_s')
            ratio = round(ts / g_ts, 3) if (ts and g_ts) else None
            P(f'| {a} | {s} | {fmt(ts)} | {fmt(ratio)} |')
    h_ts = mean([runs[f'H_wall_s{s}']['sustained_tok_s'] for s in (0, 1)
                 if runs.get(f'H_wall_s{s}')])
    penalty = round(h_ts / g_ts, 3) if (h_ts and g_ts) else None
    P(f'\n**H/G LM-loop throughput ratio = {fmt(penalty)}** '
      f'(microbench reference from e97-hetero-cma: 0.731x).')

    # --- wall-clock-matched BPB ---
    W = runs.get('H_wall_s0', {}).get('wall_seconds_budget')
    P(f'\n### WALL-CLOCK-matched held-out BPB (every arm trained {fmt(W)}s)\n')
    P('| arch | seed | tokens | wall_min | held-out BPB |')
    P('|---|---:|---:|---:|---:|')
    wall_bpb = {}
    for a in ('H', 'G', 'L'):
        bs = []
        for s in (0, 1):
            d = runs.get(f'{a}_wall_s{s}')
            if not d:
                continue
            b = bpb(d); bs.append(b)
            P(f'| {a} | {s} | {fmt(d.get("tokens"))} | {fmt(d.get("wall_minutes"))} | {fmt(b)} |')
        wall_bpb[a] = mean(bs)
    P(f'\nmean held-out BPB — H {fmt(wall_bpb.get("H"))} | G {fmt(wall_bpb.get("G"))} '
      f'| L {fmt(wall_bpb.get("L"))}')

    # --- token-matched BPB ---
    NH = mean([runs[f'H_wall_s{s}']['tokens'] for s in (0, 1)
               if runs.get(f'H_wall_s{s}')])
    P(f'\n### TOKEN-matched held-out BPB (every arm trained to N_H ~= {fmt(NH)} tokens)\n')
    P('| arch | seed | tokens | wall_min | held-out BPB |')
    P('|---|---:|---:|---:|---:|')
    tok_bpb = {}
    for a in ('H', 'G', 'L'):
        bs = []
        for s in (0, 1):
            d = runs.get(f'{a}_token_s{s}')
            if not d:
                continue
            b = bpb(d); bs.append(b)
            P(f'| {a} | {s} | {fmt(d.get("tokens"))} | {fmt(d.get("wall_minutes"))} | {fmt(b)} |')
        tok_bpb[a] = mean(bs)
    P(f'\nmean held-out BPB — H {fmt(tok_bpb.get("H"))} | G {fmt(tok_bpb.get("G"))} '
      f'| L {fmt(tok_bpb.get("L"))}')

    # --- verdict ---
    P('\n### Verdict\n')
    hv, gv = wall_bpb.get('H'), wall_bpb.get('G')
    if hv and gv:
        dw = round(hv - gv, 5)
        P(f'- WALL-clock-matched: H {hv} vs G {gv} -> delta {dw:+} BPB '
          f'({"H worse" if dw > 0 else "H better/tie"}).')
    ht, gt = tok_bpb.get('H'), tok_bpb.get('G')
    if ht and gt:
        dt = round(ht - gt, 5)
        P(f'- TOKEN-matched: H {ht} vs G {gt} -> delta {dt:+} BPB '
          f'({"H worse" if dt > 0 else "H better/tie"}).')
    P(f'- Throughput penalty paid by H: {fmt(penalty)}x gdn2-mlp '
      f'(NOT the >=0.95x task premise; the real capability head is ~0.73x).')

    txt = '\n'.join(lines)
    print(txt)
    summ = dict(param={a: runs.get(f'{a}_wall_s0', {}).get('param_report')
                       for a in ('H', 'G', 'L')},
                throughput_ratio_H_over_G=penalty, g_tok_s=g_ts, h_tok_s=h_ts,
                wall_bpb=wall_bpb, token_bpb=tok_bpb, N_H=NH, W=W,
                runs={k: (dict(bpb=bpb(v), tokens=v.get('tokens'),
                               tok_s=v.get('sustained_tok_s'),
                               wall_min=v.get('wall_minutes'),
                               stop=v.get('stop_reason'), err=v.get('error'))
                          if v else None) for k, v in runs.items()})
    json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)
    print('\nwrote', os.path.join(OUT, 'summary.json'))


if __name__ == '__main__':
    main()
