"""e97-hetero-cma: join throughput + capability + CMA into the tradeoff curve.

Reads results/{throughput,capability,cma_joint}.json and emits a single
tradeoff table (capability vs nonlinear-head fraction vs blended tok/s ratio)
plus the CMA-winner config. Pure aggregation of REAL run outputs. No mocks.
"""
import os, sys, json
_THIS = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(_THIS, 'results')


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    tput = load('throughput.json'); cap = load('capability.json'); cma = load('cma_joint.json')
    out = {}
    # throughput: best (overlap) ratio per fraction
    tmap = {}
    if tput:
        for r in tput.get('rows', []):
            if 'tok_s' not in r:
                continue
            f = int(round(r['e97_frac'] * 64))
            tmap.setdefault(f, {})[('ov' if r.get('overlap') else 'seq')] = r.get('ratio_vs_gdn2')
        out['gdn2_baseline_tok_s'] = tput.get('gdn2_baseline_tok_s')
    # capability: depth/count/recall per arm
    cmap = {}
    if cap:
        for name, s in cap.get('summary', {}).items():
            cmap[name] = s
    # tradeoff rows keyed by fraction (split_tanh arms)
    rows = []
    if cap:
        for name, s in sorted(cap['summary'].items(), key=lambda kv: kv[1].get('frac', 0)):
            if s.get('kind') not in ('linear', 'split_tanh'):
                continue
            f = s.get('frac', 0)
            t = tmap.get(f, {})
            rows.append(dict(arm=name, kind=s['kind'], frac_heads=f,
                             depth=s.get('depth'), count=s.get('count'), recall=s.get('recall'),
                             tok_s_ratio_overlap=t.get('ov'), tok_s_ratio_seq=t.get('seq')))
    out['tradeoff'] = rows
    # substrate control (gated-delta shell)
    if cap and 'shell8' in cap['summary']:
        out['substrate_control_shell8'] = cap['summary']['shell8']
    # cma winner
    if cma:
        b = cma.get('best')
        out['cma_best'] = dict(fitness=b.get('fitness'), bpb=b.get('bpb'),
                               tok_s=b.get('tok_s'), tokens=b.get('tokens'),
                               e97_heads=b['meta'].get('e97_heads'),
                               params_b=b['meta'].get('params_b'),
                               cfg={k: b['cfg'].get(k) for k in
                                    ('dim', 'lr', 'knob_lr_mult', 'lam_max', 'beta_max', 'mlp_ratio')},
                               gens_done=cma.get('gens_done'))
    json.dump(out, open(os.path.join(R, 'tradeoff.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
