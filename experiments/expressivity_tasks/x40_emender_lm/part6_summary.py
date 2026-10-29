"""Fail-closed surface accounting. No throughput values are filled for failures."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from .scale_surface import SHAPES

ROOT=Path(__file__).resolve().parent/'receipts/part6'


def main():
    rows=[];arms=[]
    for scale in SHAPES:
        for t in (512,2048):
            for b in (64,128):
                pair={}
                for arm in ('pure','transformer'):
                    path=ROOT/'surface'/f'{scale}_{arm}_B{b}_T{t}.json'
                    result=json.loads(path.read_text()) if path.exists() else dict(status='missing')
                    qualified=result['status']=='passed' and result.get('measured_updates',0)>=20 and len(result.get('seconds',[]))>=20
                    cell=dict(scale=scale,context=t,batch=b,arm=arm,status='passed' if qualified else result['status'],
                              sustained=qualified,microbatch=result.get('microbatch'),checkpoint_input_offload=result.get('checkpoint_input_offload',False),allocator_expandable=result.get('allocator_expandable',False),tokens_per_second=result.get('tokens_per_second') if qualified else None,
                              peak_allocated=result.get('peak_allocated'),peak_reserved=result.get('peak_reserved'))
                    if cell['status']=='passed' and not qualified:cell['status']='insufficient_samples'
                    pair[arm]=cell;arms.append(cell)
                ratio=pair['pure']['tokens_per_second']/pair['transformer']['tokens_per_second'] if all(x['sustained'] for x in pair.values()) else None
                rows.append(dict(scale=scale,context=t,batch=b,ratio_to_transformer=ratio,pure=pair['pure'],transformer=pair['transformer']))
    curves=ROOT/'curve/summary.json'
    curve=json.loads(curves.read_text()) if curves.exists() else dict(status='missing')
    production={f'{t}{tag}':json.loads((ROOT/f'production_T{t}{tag}.json').read_text())['status'] if (ROOT/f'production_T{t}{tag}.json').exists() else 'missing' for t in (512,2048) for tag in ('','_expandable')}
    byte_gates={}
    for context in (512,2048):
        for batch in (1,32):
            for precision in ('','_fp32'):
                tag='_expandable' if batch==32 else ''
                path=ROOT/f'checkpoint_gate_B{batch}_T{context}_packed{precision}{tag}.json'
                gate=json.loads(path.read_text()) if path.exists() else dict(status='missing')
                byte_gates[path.name]='passed' if gate['status']=='passed' and gate.get('equality','').startswith('raw contiguous bytes') else 'not-qualified'
    fixtures={}
    for suite,expected in (('x31_tape_cell_fused',109),('x36_tape_perf',70)):
        path=ROOT/f'{suite}.xml'
        counts=dict(tests=0,failures=0,errors=0,skipped=0)
        if path.exists():
            for node in ET.parse(path).getroot().iter('testsuite'):
                for key in counts:counts[key]+=int(node.get(key,0))
        fixtures[suite]=dict(status='passed' if counts['tests']-counts['skipped']>=expected and counts['failures']==counts['errors']==0 else 'not-qualified',**counts)
    local=(ROOT/'local_tests.log').read_text() if (ROOT/'local_tests.log').exists() else ''
    passed=re.findall(r'(\d+) passed',local)
    local_tests=dict(status='passed' if passed and int(passed[-1])>=15 and not re.search(r'\d+ (failed|error)',local) else 'not-qualified',passed=int(passed[-1]) if passed else 0)
    certifications=all(v=='passed' for v in production.values()) and all(v=='passed' for v in byte_gates.values()) and all(v['status']=='passed' for v in fixtures.values()) and local_tests['status']=='passed'
    pure=[a for a in arms if a['arm']=='pure' and a['sustained']]
    compiler_tests={}
    for registers in (64,128):
        items=[]
        for suite in ('x31_tape_cell_fused','x36_tape_perf'):
            path=ROOT/f'registers{registers}_{suite}_tests.json'
            items.append(json.loads(path.read_text())['status'] if path.exists() else 'missing')
        compiler_tests[str(registers)]='passed' if all(v=='passed' for v in items) else 'not-qualified'
    fusion_tests=[]
    for suite in ('x31_tape_cell_fused','x36_tape_perf'):
        path=ROOT/f'fused_pool_{suite}_tests.json'
        fusion_tests.append(json.loads(path.read_text())['status'] if path.exists() else 'missing')
    fusion_certified=all(v=='passed' for v in fusion_tests)
    pilots=[]
    for label in ('pilot_B128','pilot_head_stream_B64','pilot_head_stream_B128','pilot_registers128','pilot_registers64'):
        wall=ROOT/label/'wall.json'
        if not wall.exists():continue
        result=json.loads(wall.read_text())
        candidate_curve=ROOT/'head_stream_curve/summary.json' if 'head_stream' in label else ROOT/'curve/summary.json'
        if label.startswith('pilot_registers'):candidate_curve=ROOT/(label.removeprefix('pilot_')+'_curve')/'summary.json'
        matched=candidate_curve.exists() and json.loads(candidate_curve.read_text())['status']=='passed'
        if result['status']=='passed' and len(result.get('seconds',[]))>=20 and matched and (not result.get('maxnreg') or compiler_tests[str(result['maxnreg'])]=='passed'):
            pilots.append(dict(scale='pilot-v1',label=label,arm='pure',status='passed',sustained=True,
                               batch=result['batch'],context=result['context'],tokens_per_second=result['tokens_per_second'],
                               peak_allocated=result['peak_allocated'],peak_reserved=result['peak_reserved']))
    unqualified=[]
    for path in sorted(ROOT.glob('pilot_B*_registers*.json')):
        result=json.loads(path.read_text())
        matched=ROOT/('matched_'+path.stem)/'summary.json'
        quality=json.loads(matched.read_text()) if matched.exists() else dict(status='missing')
        valid=result['status']=='passed' and len(result.get('seconds',[]))>=20
        item=dict(scale='pilot-v1',label=path.stem,arm='pure',status=result['status'],sustained=valid,
                  batch=result['batch'],context=result['context'],tokens_per_second=result.get('tokens_per_second'),
                  peak_allocated=result.get('peak_allocated'),peak_reserved=result.get('peak_reserved'),
                  matched_curve_status=quality['status'],checkpoint=result.get('checkpoint'),head_streams=result.get('head_streams'),
                  maxnreg=result.get('registers',0),loss_chunk=result.get('loss_chunk',1024),deferred_pack=result.get('deferred_pack'),packed=result.get('packed'),fused_pool=result.get('fused_pool',False))
        if valid and (not result.get('fused_pool') or fusion_certified) and quality['status']=='passed' and quality.get('tokens',0)>=2_000_000 and (not result.get('registers') or compiler_tests[str(result['registers'])]=='passed'):pilots.append(item)
        else:unqualified.append(item)
    qualified=pure+pilots
    best=max(qualified,key=lambda a:a['tokens_per_second']) if qualified else None
    complete=all(a['sustained'] for a in arms)
    result=dict(status='passed' if complete and best and best['tokens_per_second']>=100000 and curve['status']=='passed' and certifications else 'failed',
                surface_status='passed' if complete and certifications else 'failed',surface_measurement_status='passed' if complete else 'failed',sustained_arms=sum(a['sustained'] for a in arms),expected_arms=24,
                matched_ratios=sum(r['ratio_to_transformer'] is not None for r in rows),expected_ratios=12,
                surface=rows,pilots=pilots,unqualified_pilots=unqualified,best_sustained_pure=best,production_parity=production,checkpoint_byte_parity=byte_gates,native_fixtures=fixtures,compiler_fixtures=compiler_tests,fused_pool_fixtures=fusion_tests,local_tests=local_tests,curve=curve,
                irreducible_floor_established=False,
                residual_risks=['Full performance qualification requires all surface arms, exact production parity, and matched training curve.',
                                'Journal packing preserves transient writer reservation and adds host synchronization.',
                                'FP32 bounded projection is a new matched probe protocol; part5 dense bf16 ratios are separate evidence.'])
    if best:
        mean=best['batch']*best['context']/best['tokens_per_second'];budget=best['batch']*best['context']/100000
        result['arithmetic']=dict(mean_seconds=mean,gate_budget_seconds=budget,required_savings_seconds=max(0,mean-budget),required_speedup=mean/budget)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','sustained_arms','matched_ratios','best_sustained_pure')},indent=2))


if __name__=='__main__':main()
