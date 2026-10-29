"""Fail-closed synthesis; component experiments never certify production."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def load(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else dict(status='missing')

def main():
    result=dict(status='failed',production_changes=False,production_speedup_accepted=False,
      acceptance_targets={'one_x':'not verified','pilot_100k':'failed; historical qualified 54836.10 tokens/s'},
      profiles=[],cache_analysis=[],read_probe=load('read_probe.json')['status'],cache_probe=load('cache_probe.json')['status'])
    for context in (512,2048):
        pure=load(f'profile_pure_T{context}.json');control=load(f'profile_transformer_T{context}.json')
        row=dict(context=context,pure_status=pure['status'],transformer_status=control['status'])
        if pure['status']==control['status']=='passed':
            assert pure['batch']==control['batch']==64
            for field in ('checkpoint','offload','microbatch'):assert pure['envelope'][field]==control['envelope'][field]
            assert abs(pure['parameters']['total']/control['parameters']['total']-1)<.001
            row.update(pure_seconds=pure['wall_seconds'],transformer_seconds=control['wall_seconds'],diagnostic_step_ratio=pure['wall_seconds']/control['wall_seconds'],exclusive_seconds=pure['exclusive_seconds'],scope='synchronized diagnostic only; not sustained ratio')
            assert abs(pure['unattributed_seconds'])<.1
        result['profiles'].append(row)
        # Current scale uses 24 blocks, physical B32, two microbatches serially.
        s2=min(736,context);s3=min(1472,context)
        cache_bytes=32*24*context*(s2+s3)*4
        outputs=32*24*context*2*128*4
        result['cache_analysis'].append(dict(context=context,heads=[2,3],layers=24,physical_batch=32,effective_batch=64,
            probability_cache_bytes=cache_bytes,probability_cache_GiB=cache_bytes/2**30,output_only_cache_bytes=outputs,
            output_only_cache_GiB=outputs/2**30,
            interpretation='Outputs alone cannot replace score/probability replay. Probability cache sizes are storage accounting, not measured full-model peak or speedup. Requires recording during initial checkpoint pass; one microbatch live at a time.'))
    curves=[load(f'curve_{arm}.json') for arm in ('fp32','bf16')]
    result['curve_status']=[r['status'] for r in curves]
    result['sustained_ratios']=[]
    for context in (512,2048):
        arms=[load(f'sustained/attempts/1.3B_{arm}_B64_T{context}/00.json') for arm in ('pure','transformer')]
        if all(r['status']=='passed' for r in arms):
            for r in arms:assert r['warmups']==3 and r['measured_updates']==20 and len(r['seconds'])==20
            for field in ('microbatch','checkpoint','offload'):assert arms[0]['envelope'][field]==arms[1]['envelope'][field]
            result['sustained_ratios'].append(dict(context=context,batch=64,microbatch=32,pure_tok_s=arms[0]['tokens_per_second'],transformer_tok_s=arms[1]['tokens_per_second'],pure_time_over_transformer=arms[1]['tokens_per_second']/arms[0]['tokens_per_second'],physical_gpus=[r['physical_gpu'] for r in arms],scope='unchanged production, fixed shared envelope, 3 warmups/20 updates; not an optimization before/after'))
    pilot=load('pilot/fresh_gate.json')
    result['pilot']=dict(measurement_status=pilot['status'])
    if pilot['status']=='passed':
        assert len(pilot['seconds'])==20 and pilot['warmups']==3
        result['pilot'].update(tokens_per_second=pilot['tokens_per_second'],gate='passed' if pilot['tokens_per_second']>=100000 else 'failed',curve='inherited unchanged part8 B128 matched curve')
        result['acceptance_targets']['pilot_100k']=result['pilot']['gate']
    if all(r['status']=='passed' for r in curves):
        assert curves[0]['data_sha256']==curves[1]['data_sha256']
        rows=[[json.loads(l) for l in (OUT/f'curve_{arm}.jsonl').read_text().splitlines()] for arm in ('fp32','bf16')]
        assert len(rows[0])==len(rows[1])==153
        differences=[abs(a['loss']-b['loss']) for a,b in zip(*rows)]
        result['curve_comparison']=dict(tokens=curves[0]['tokens'],steps=153,max_abs_loss_difference=max(differences),mean_abs_loss_difference=sum(differences)/len(differences),final_fp32_loss=rows[0][-1]['loss'],final_bf16_loss=rows[1][-1]['loss'],final_absolute_loss_difference=differences[-1],matched_data=True,scope='pilot; candidate BF16 forward with FP32 surrogate replay; does not certify correctly redefined BF16 adjoints or scale quality')
    result['missing_acceptance']=[
      'Separate wall attribution of reverse read mathematics versus sparse adjoints within the fused kernel.',
      'BF16 candidate CPU/native bitwise parity and explicitly redefined adjoint certification; current experiment uses FP32 surrogate replay.',
      'No production optimization landed; no after-optimization speedup or 1x claim. Fresh unchanged sustained measurements are listed separately.',
      'Full-step 1.3B hybrid caching implementation and measured memory/step time; component experiment only.',
      'Trained-density event suppression exactness and sustained speedup.']
    result['rate_class_probe']=load('rate_probe.json')['status']
    result['store_probe']=load('store_probe.json')['status']
    result['batch_read_probe']=load('batch_read_probe.json')['status']
    result['reverse_cycle_diagnostics']=[]
    for t in (512,2048):
        cycles=load(f'reverse_clock/cycles_T{t}.json')
        result['reverse_cycle_diagnostics'].append(dict(context=t,status=cycles['status'],cycle_fractions=cycles.get('cycle_fractions',{}),cycle_weighted_device_ms_estimates=cycles.get('cycle_weighted_device_ms_estimates',{}),scope='instrumented occupancy-weighted diagnostic; not additive wall phases'))
    for row in result['cache_analysis']:
        actual=[r for r in load('read_probe.json').get('rows',[]) if r['context']==row['context'] and r['head'] in (2,3)]
        if len(actual)==2:assert sum(r['probability_cache_bytes_per_lane'] for r in actual)*32*24==row['probability_cache_bytes']
    result['production_bindings']=[]
    for context in (512,2048):
        gate=json.loads((ROOT/f'receipts/part8/production_T{context}.json').read_text())
        fresh=json.loads((ROOT/f'receipts/part9/production_T{context}.json').read_text())
        assert fresh['status']=='passed' and fresh['pressure_gate']
        assert gate['status']=='passed' and gate['pressure_gate']
        for name in ('model.py','forward.py','backward.py','kernels.py','scale_surface.py'):
            assert gate['sources'][name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        result['production_bindings'].append(dict(context=context,pressure_gate=True,source='part8 source-bound certification plus part9 fresh pressure gate',source_receipt_sha256=hashlib.sha256((ROOT/f'receipts/part8/production_T{context}.json').read_bytes()).hexdigest(),part9_receipt_sha256=hashlib.sha256((ROOT/f'receipts/part9/production_T{context}.json').read_bytes()).hexdigest()))
    result['bf16_certification']=dict(status='failed',cpu_native_prefix_bitwise_failures=sum(not r['cpu_prefix_bitwise_equal'] for r in load('read_probe.json').get('rows',[])),prefix_cases=len(load('read_probe.json').get('rows',[])),adjoint_certification='not run; FP32 surrogate replay retained',production_adoption=False)
    result['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('part10_*.py')}
    (OUT/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
