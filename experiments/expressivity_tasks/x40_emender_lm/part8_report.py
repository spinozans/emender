"""Fail-closed v2 surface report; missing and failed arms stay visible."""
import json
from .part8_surface import ROOT,OUT,ARMS,SHAPES


def main():
    rows=[];ratios=[];problems=[]
    text=['# X40 part 8 — three-architecture throughput surface v2','',
          'Synthetic capacity-matched training: scale labels name pure trainable count; comparators match pure **total** parameters within 0.1%. This is a compute experiment, not a scale quality comparison.', '',
          'Baselines use bf16 dense compute and fused bf16 vocabulary projection with fp32 master parameters/AdamW. Pure uses bf16 dense blocks, its parity-certified **fp32 native store** and bounded fp32 projection. This precision asymmetry is intentional; ratios compare against best-practice baselines.', '',
          'Every rate requires three warmups and 20 synchronized forward/backward/clip/AdamW updates. Each arm starts bare. Only a measured CUDA OOM permits envelope escalation. OOM peak is a partial-run lower bound, never required capacity. First fitting means minimal in the declared policy ordering, not a globally fastest combination.', '',
          'Failed envelopes remain in attempts/. The missing T512 checkpoint-gate filename caused a preflight failure, preserved in preflight_failures/ and retried using the existing strict B32 gate. Initial GDN2 timings with the forced pure backbone are rejected as best-practice evidence and retained in rejected_forced_gdn_backbone/. Accepted GDN2 configurations instead tune width/depth near the registered recipe MLP ratio; configs.json receipts their exact differences from the paper configuration. A partial packed B32 pure trial was interrupted when policy review found the missing checkpoint-only B32 candidate; its unqualified log remains in policy_corrections/. Each smaller microbatch now tests checkpoint-only before adding packing.', '',
          '| Scale | T | B | Architecture | Status | Sustained tok/s | Peak allocated GiB | Envelope |',
          '|---|---:|---:|---|---|---:|---:|---|']
    for scale in SHAPES:
        for context in (512,2048):
            for batch in (64,128):
                points={}
                for arm in ARMS:
                    name=f'{scale}_{arm}_B{batch}_T{context}'
                    path=OUT/'surface'/f'{name}.json'
                    r=json.loads(path.read_text()) if path.exists() else dict(status='missing',scale=scale,context=context,batch=batch,arm=arm)
                    if r['status']=='passed':
                        assert len(r['seconds'])==20 and r['warmups']==3
                        assert abs(r['parameters']['total']/r['pure_parameters']['total']-1)<.001
                        assert abs(r['tokens_per_second']-batch*context*20/sum(r['seconds']))<1e-6
                        attempts=[json.loads((ROOT/p).read_text()) for p in r['attempt_receipts']]
                        assert attempts[0]['envelope']['name']=='bare'
                        assert all(x['status']=='oom' for x in attempts[:-1])
                        assert attempts[-1]['status']=='passed'
                        if arm=='gdn2':
                            config=next(c for c in json.loads((OUT/'configs.json').read_text())['configs'] if c['scale']==scale)
                            assert r['parameters']==config['gdn2_parameters']
                            for key in ('width','depth','mlp_hidden'):
                                assert r['gdn2_config'][key]==config['gdn2_'+key]
                        points[arm]=r['tokens_per_second']
                    else:problems.append(f'{name}: {r["status"]} {r.get("error","")}')
                    rate=f'{r["tokens_per_second"]:,.1f}' if r['status']=='passed' else '—'
                    peak=f'{r["peak_allocated"]/2**30:.3f}' if 'peak_allocated' in r else '—'
                    text.append(f'| {scale} | {context} | {batch} | {arm} | {r["status"]} | {rate} | {peak} | {r.get("envelope",{}).get("name","—")} |')
                    rows.append(r)
                ratios.append(dict(scale=scale,context=context,batch=batch,
                    pure_over_transformer=points['pure']/points['transformer'] if {'pure','transformer'}<=points.keys() else None,
                    pure_over_gdn2=points['pure']/points['gdn2'] if {'pure','gdn2'}<=points.keys() else None,
                    gdn2_over_transformer=points['gdn2']/points['transformer'] if {'gdn2','transformer'}<=points.keys() else None))
    text+=['','| Scale | T | B | Pure / transformer | Pure / GDN2 | GDN2 / transformer |','|---|---:|---:|---:|---:|---:|']
    for r in ratios:
        vs=['—' if r[k] is None else f'{r[k]:.4f}' for k in ('pure_over_transformer','pure_over_gdn2','gdn2_over_transformer')]
        text.append(f'| {r["scale"]} | {r["context"]} | {r["batch"]} | '+ ' | '.join(vs)+' |')
    text+=['','The 100k gate requires fresh sustained bare pilot throughput, pressure-inclusive production parity and a matched curve. It uses the original v1 pilot (about 21.719M trainable), whereas the surface uses the parameter-targeted scale family (21.999M at 22M). Their bare rates are separate results; the historical 54.8k pilot rate is not substituted for a surface cell. No historical measurement substitutes for a missing new gate. See gate.json for qualification.', '',
           'Residual risks: synthetic capacity matching does not establish model quality across scale; B1 native parity does not establish byte equality of every full scale model. Envelope ordering does not test every combination. Sparse backward union pooling and event suppression retain exact replay; no unsupported skipping of value-consuming events is accepted. A failed 100k measurement establishes no hardware floor.', '',
           'No WG, subagents, .pt artifacts, Frontier, Slurm or resilient DiLoCo changes. AGENTS.md and CLAUDE.md are unchanged. contact_supervisor was unavailable. Protected GPU cards and the depth probe were untouched.']
    gatepath=OUT/'gate.json';gate=json.loads(gatepath.read_text()) if gatepath.exists() else dict(status='missing')
    trends=[]
    for context in (512,2048):
        for batch in (64,128):
            curve=[r for r in ratios if r['context']==context and r['batch']==batch]
            item=dict(context=context,batch=batch)
            for key in ('pure_over_transformer','pure_over_gdn2'):
                values=[r[key] for r in curve]
                item[key]=dict(values=values,monotonically_increasing=all(v is not None for v in values) and all(a<=b for a,b in zip(values,values[1:])))
            trends.append(item)
    text+=['','Measured scale trends (22M → 150M → 1.3B):']
    for trend in trends:
        for key in ('pure_over_transformer','pure_over_gdn2'):
            value=trend[key]
            sequence=' → '.join('missing' if v is None else f'{v:.4f}' for v in value['values'])
            text.append(f'- T{trend["context"]}/B{trend["batch"]}, {key}: {sequence}; monotonic increase: {value["monotonically_increasing"]}.')
    text+=['','Uniform monotonic improvement against both comparators is established only if every corresponding trend above passes. A reversal is retained as an experimental result, not averaged away. These are measured throughput ratios, not statistical significance or model-quality claims.']
    text[2:2]=[f'Sustained surface: **{sum(r["status"]=="passed" for r in rows)}/36** arms. 100k acceptance gate: **{gate["status"].upper()}**.', '']
    text+=['',f'Fresh 100k gate: **{gate["status"].upper()}**. Sustained tok/s: {gate.get("tokens_per_second", "unmeasured")}. Qualified parity/curve measurement: {gate.get("qualified_measurement", False)}.',
           f'Matched curve: `{json.dumps(gate.get("curve", {}), sort_keys=True)}`.',
           '', 'Ratio curves: [SVG](receipts/part8/ratio_curves.svg). Full configurations and external kernel provenance: [configs.json](receipts/part8/configs.json). Each surface row links its complete ordered attempt receipts in the corresponding JSON under receipts/part8/surface/.']
    summary=dict(status='passed' if not problems and gate['status']=='passed' else 'failed',surface_status='passed' if not problems else 'failed',surface_arms=len(rows),successful_arms=sum(r['status']=='passed' for r in rows),ratios=ratios,scale_trends=trends,failures=problems,gate=gate)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'REPORT_PART8.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('ratios','gate')},indent=2))

if __name__=='__main__':main()
