"""Receipt-bound part-five acceptance; absent or failed cells stay failed."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
R=ROOT/'receipts/part5'


def main():
    rows=[]
    for scale in ('22M','150M','1.3B'):
        for t in (512,2048):
            for b in (64,128):
                pair={}
                for arm in ('pure','transformer'):
                    p=R/'surface'/f'{scale}_{arm}_B{b}_T{t}.json'
                    pair[arm]=json.loads(p.read_text()) if p.exists() else {'status':'not-run'}
                good=all(v['status']=='passed' for v in pair.values())
                rows.append(dict(scale=scale,batch=b,context=t,status='passed' if good else 'failed',
                    pure_status=pair['pure']['status'],transformer_status=pair['transformer']['status'],
                    pure_tokens_per_second=pair['pure'].get('tokens_per_second'),transformer_tokens_per_second=pair['transformer'].get('tokens_per_second'),
                    ratio_to_transformer=pair['pure']['tokens_per_second']/pair['transformer']['tokens_per_second'] if good else None))
    levers={}
    labels={'pilot_baseline','pilot_reads','pilot_backward','pilot_B128','pilot_B256','pilot_B128_release','pilot_fused'}
    labels.update(p.name for p in R.glob('pilot_*') if p.is_dir() and p.name!='pilot_reads_cpu')
    for label in sorted(labels):
        p=R/label/'wall.json';failure=R/label/'failure.json'
        levers[label]=json.loads(p.read_text()) if p.exists() else (json.loads(failure.read_text()) if failure.exists() else {'status':'not-run'})
    curve=R/'curve/summary.json';curve=json.loads(curve.read_text()) if curve.exists() else {'status':'not-run'}
    bounded_path=R/'curve_bounded/summary.json'
    bounded_curve=json.loads(bounded_path.read_text()) if bounded_path.exists() else {'status':'not-run'}
    eligible={name:value for name,value in levers.items() if not name.endswith('_pre_guard') and value.get('status')=='passed' and value.get('updates',0)>=20 and (bounded_curve if value.get('bounded_projection') else curve).get('status')=='passed' and (bounded_curve if value.get('bounded_projection') else curve).get('tokens',0)>=2000000}
    best=max(eligible,key=lambda name:eligible[name]['tokens_per_second']) if eligible else None
    achieved=[v for v in eligible.values() if v.get('tokens_per_second',0)>=100000 and v.get('batch',0)>=64 and v.get('updates',0)>=20]
    gates={}
    for name in ('production_parity','production_T2048','reads_parity','backward_parity','fused_parity','fused_parity_after_guard'):
        path=R/(name+'.json');gates[name]=json.loads(path.read_text()).get('status','not-run') if path.exists() else 'not-run'
    arithmetic=[]
    for name,value in levers.items():
        if value.get('status')=='passed':
            budget=value['batch']*value['context']/100000
            mean=sum(value['seconds'])/len(value['seconds'])
            arithmetic.append(dict(lever=name,mean_seconds=mean,gate_budget_seconds=budget,
                required_speedup=100000/value['tokens_per_second'],required_savings_seconds=mean-budget))
    result=dict(status='passed' if achieved and all(r['status']=='passed' for r in rows) and all(v=='passed' for v in gates.values()) and curve.get('status')=='passed' and curve.get('tokens',0)>=2000000 else 'failed',
        best_qualified_lever=best,best_tokens_per_second=eligible[best]['tokens_per_second'] if best else None,
        interim_reads=dict(threshold=60000,status='passed' if levers['pilot_reads'].get('tokens_per_second',0)>=60000 else 'failed',tokens_per_second=levers['pilot_reads'].get('tokens_per_second')),
        interim_backward=dict(threshold=85000,status='passed' if best and eligible[best]['tokens_per_second']>=85000 else 'failed',best_tokens_per_second=eligible[best]['tokens_per_second'] if best else None),
        residual_risks=['18/24 scale arms OOM; 11/12 ratios are missing.',
            '100k and both interim gates remain unmet; no irreducible floor established.',
            'Nonrecording batched reads fail closed after the reverted checkpoint heap assertion; cause remains open.',
            'Full inclusive launch fusion and replay sizing synchronization removal remain incomplete.',
            'Scale names count pure trainable weights; controls match total weights, not trainable weights.'],
        throughput_gate='passed' if achieved else 'failed',bounded_curve=bounded_curve,certifications=gates,arithmetic=arithmetic,
        gate=dict(training_tokens_per_second=100000,minimum_batch=64,contexts=[512,2048],matched_tokens=2000000),
        scale_surface=rows,levers=levers,curve=curve,
        scope='No throughput from an OOM or missing cell; no irreducible floor asserted. Surface names pure trainable count; control matches total including frozen features.')
    (R/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],surface_pairs_passed=sum(r['status']=='passed' for r in rows),surface_pairs=12),indent=2))


if __name__=='__main__':main()
