"""Aggregate the redo-e99-1-3b raw artifacts into the report tables.

Reads:
  results/run1/all_results.json        (LM screen: anchors + CMA)
  results/capability/capability_summary.json  (capability axis)
Writes:
  results/report_tables.md  (markdown tables used in E99_MIXTURE_AWARE_LM_CMA.md)
  results/report_tables.json
Prints the same to stdout. Pure aggregation of REAL run outputs; no fabrication.
"""
import os, sys, json, math

_THIS = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(_THIS, 'results', 'run1', 'all_results.json')
CAP = os.path.join(_THIS, 'results', 'capability', 'capability_summary.json')
OUT_MD = os.path.join(_THIS, 'results', 'report_tables.md')
OUT_JSON = os.path.join(_THIS, 'results', 'report_tables.json')


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def lm_rows(run):
    """Flatten anchors + cma into uniform rows."""
    rows = []
    for name, r in (run.get('anchors') or {}).items():
        rows.append(_lmrow(name, r, 'anchor'))
    for r in (run.get('cma') or []):
        rows.append(_lmrow(r.get('name'), r, 'cma'))
    return [x for x in rows if x]


def _lmrow(name, r, kind):
    if not r or 'avg_loss' not in r:
        return None
    hd = r.get('heldout') or {}
    c = r.get('counts') or {}
    rt = r.get('roundtrip') or {}
    return dict(name=name, kind=kind,
                n_gdn=c.get('gdn2_recall'), n_nonlin=c.get('nonlin'),
                n_shell=c.get('gdn2_nonlin_shell'), n_track=c.get('e97_track'),
                n_count=c.get('count'), n_latch=c.get('latch'),
                avg_loss=r.get('avg_loss'), final_loss=r.get('final_loss'),
                bpb=hd.get('heldout_bpb'), tok_s=r.get('tok_s'),
                steps=r.get('steps'), tokens=r.get('tokens'),
                params_b=r.get('params_b'), wall_min=r.get('wall_minutes'),
                nan=r.get('nan_seen'), rt_ok=rt.get('ok'), gpu=r.get('gpu'),
                shell_nonlin=r.get('shell_state_nonlin'))


def fmt(v, n=4):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.{n}f}'
    return str(v)


def main():
    run = load(RUN)
    cap = load(CAP)
    md = ['# redo-e99-1-3b — aggregated tables (real run outputs)\n']
    out = {}

    if run:
        rows = lm_rows(run)
        rows_sorted = sorted(rows, key=lambda x: (x['avg_loss'] if x['avg_loss'] is not None else 1e9))
        out['lm_rows'] = rows
        md.append(f"## LM screen — {run.get('n_evals')} evals, "
                  f"{run.get('wall_minutes_per_eval')}-min bf16, GPUs {run.get('gpus_used')}, "
                  f"{run.get('aggregate_gpu_minutes')} GPU-min\n")
        md.append("Ranked by AvgLoss (CMA fitness, lower=better). BPB on canonical Pile held-out slice.\n")
        md.append("| rank | name | kind | gdn/nonlin/shell | AvgLoss | Final | held-out BPB | tok/s | steps | params_B | NaN | RT |")
        md.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|")
        for i, x in enumerate(rows_sorted, 1):
            md.append(f"| {i} | {x['name']} | {x['kind']} | "
                      f"{x['n_gdn']}/{x['n_nonlin']}/{x['n_shell']} | "
                      f"{fmt(x['avg_loss'])} | {fmt(x['final_loss'])} | {fmt(x['bpb'])} | "
                      f"{fmt(x['tok_s'],0)} | {x['steps']} | {fmt(x['params_b'],3)} | "
                      f"{'Y' if x['nan'] else 'n'} | {'ok' if x['rt_ok'] else ('—' if x['rt_ok'] is None else 'FAIL')} |")
        md.append('')

        # three-way matched-fraction triples (a) dense / (b) shell / (c) nonlin
        byname = {x['name']: x for x in rows}
        md.append("## Three-way fairness control — matched-fraction triples\n")
        md.append("(a) native GDN-2 linear = M0_dense (same at every f). (b) GDN-2-shell nonlinear = S*. "
                  "(c) legacy UnifiedCell nonlin = C*. (a)vs(b)=nonlinearity itself; (b)vs(c)=external plumbing.\n")
        md.append("| f (nonlin slot) | (a) dense AvgLoss/BPB | (b) shell AvgLoss/BPB | (c) nonlin-corner AvgLoss/BPB | (b)−(a) BPB | (b)−(c) BPB |")
        md.append("|---|---|---|---|---:|---:|")
        triples = [('1/6', 'M0_dense_gdn2', 'S1_gdn_shell_f17', 'C1_gdn_nonlin_f17'),
                   ('1/3', 'M0_dense_gdn2', 'S2_gdn_shell_f33', 'C2_gdn_nonlin_f33'),
                   ('1/2', 'M0_dense_gdn2', 'S3_gdn_shell_f50', 'C3_gdn_nonlin_f50')]
        tri_out = []
        for f, a, b, c in triples:
            A, B, C = byname.get(a), byname.get(b), byname.get(c)
            if not (A and B and C):
                continue
            d_ba = (B['bpb'] - A['bpb']) if (B['bpb'] and A['bpb']) else None
            d_bc = (B['bpb'] - C['bpb']) if (B['bpb'] and C['bpb']) else None
            md.append(f"| {f} | {fmt(A['avg_loss'])} / {fmt(A['bpb'])} | "
                      f"{fmt(B['avg_loss'])} / {fmt(B['bpb'])} | "
                      f"{fmt(C['avg_loss'])} / {fmt(C['bpb'])} | "
                      f"{fmt(d_ba)} | {fmt(d_bc)} |")
            tri_out.append(dict(f=f, a=A, b=B, c=C, d_ba_bpb=d_ba, d_bc_bpb=d_bc))
        out['triples'] = tri_out
        md.append('')

    if cap:
        s = cap['summary']
        out['capability'] = s
        md.append(f"## Capability axis — {cap.get('steps')} steps, seeds {cap.get('seeds')}, "
                  f"depth{cap.get('depth')}/h{cap.get('n_heads')}/n{cap.get('n_state')}, shell={cap.get('shell_nonlin')}\n")
        probes = cap.get('eval_ts') and list(next(iter(s.values()))['per_probe'].keys())
        md.append("| mixture | gdn/nonlin/shell | mean | min | " + " | ".join(p.split('_')[0] for p in probes) + " |")
        md.append("|---|---|---:|---:|" + "---:|" * len(probes))
        for name, v in s.items():
            c = v['counts']
            pp = v['per_probe']
            md.append(f"| {name} | {c.get('gdn2_recall')}/{c.get('nonlin')}/{c.get('gdn2_nonlin_shell')} | "
                      f"{fmt(v['mean'])} | {fmt(v['min'])} | " +
                      " | ".join(fmt(pp[p], 3) for p in probes) + " |")
        md.append('')

    text = '\n'.join(md)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, 'w').write(text)
    json.dump(out, open(OUT_JSON, 'w'), indent=2, default=str)
    print(text)
    print(f"\n[wrote] {OUT_MD}\n[wrote] {OUT_JSON}")


if __name__ == '__main__':
    main()
