"""Validate receipt consistency; a passed audit does not pass the task gate."""
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def load(name):return json.loads((OUT/name).read_text())

def main():
    gate=load('gate.json');assert gate['status']=='failed' and not gate['production_changes']
    assert not gate['production_speedup_accepted']
    commands=[row.split('\t',2) for row in (OUT/'commands.tsv').read_text().splitlines()]
    assert len(commands)==18 and len({r[0] for r in commands})==18
    source=hashlib.sha256((OUT/'profile_source.py').read_bytes()).hexdigest()
    initial_source=hashlib.sha256((OUT/'profile_initial_source.py').read_bytes()).hexdigest()
    for context in (512,2048):
        initial=load(f'profile_initial_pure_T{context}.json')
        assert initial['sources']['part10_profile.py']==initial_source
        for arm in ('pure','transformer'):
            p=load(f'profile_{arm}_T{context}.json');assert p['status']=='passed'
            assert p['sources']['part10_profile.py']==source
            assert abs(sum(p['exclusive_seconds'].values())+p['unattributed_seconds']-p['wall_seconds'])<1e-8
            assert all(v>=0 for v in p['exclusive_seconds'].values())
            s=load(f'sustained/attempts/1.3B_{arm}_B64_T{context}/00.json')
            assert s['status']=='passed' and len(s['seconds'])==len(s['losses'])==20 and s['warmups']==3
            expected=s['batch']*s['context']*20/sum(s['seconds'])
            assert math.isclose(expected,s['tokens_per_second'],rel_tol=1e-12)
    curves=[load(f'curve_{arm}.json') for arm in ('fp32','bf16')]
    assert curves[0]['data_sha256']==curves[1]['data_sha256']
    for arm,r in zip(('fp32','bf16'),curves):
        assert r['status']=='passed' and r['tokens']==5013504 and r['steps']==153
        rows=[json.loads(line) for line in (OUT/f'curve_{arm}.jsonl').read_text().splitlines()]
        assert len(rows)==153
        for i,row in enumerate(rows):assert row['tokens']==(i+1)*32768 and math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
    assert gate['bf16_certification']['status']=='failed'
    assert gate['bf16_certification']['cpu_native_prefix_bitwise_failures']==gate['bf16_certification']['prefix_cases']==8
    assert len(load('cache_probe.json')['rows'])==8
    assert all(r['query_adjoint_bitwise'] for r in load('cache_probe.json')['rows'])
    for name in ('store_probe','rate_probe','batch_read_probe'):
        result=load(name+'.json')
        if result['status']=='passed':assert len(result['rows'])==8
    for context in (512,2048):
        clock=load(f'reverse_clock/cycles_T{context}.json')
        if clock['status']=='passed':
            parity=load(f'reverse_clock/parity_T{context}.json');assert parity['status']=='passed' and parity['pressure_gate']
            assert len(clock['raw'])==192 and math.isclose(sum(clock['cycle_fractions'].values()),1.,rel_tol=1e-12)
            assert load(f'reverse_clock/profile_pure_T{context}.json')['loss']==load(f'profile_pure_T{context}.json')['loss']
    assert '3 passed' in (OUT/'cpu_tests.log').read_text()
    for name,sha in gate['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for name in ('part10_clock_kernel_generated.py','part10_ieee_read.py'):
        measured=gzip.decompress((OUT/'measured_sources'/(name+'.gz')).read_bytes())
        assert measured.rstrip()+b'\n'==(ROOT/name).read_bytes()
        if name=='part10_clock_kernel_generated.py':
            for context in (512,2048):assert load(f'reverse_clock/cycles_T{context}.json')['instrumented_sha256']==hashlib.sha256(measured).hexdigest()
        else:assert load('rate_probe.json')['sources'][name]==hashlib.sha256(measured).hexdigest()
    assert not list(OUT.rglob('*.pt'))
    result=dict(status='passed',meaning='receipt consistency only; task acceptance remains failed',command_exit_codes={r[0]:int(r[1]) for r in commands},cpu_tests='3 passed',paired_curve_tokens_per_arm=5013504,sustained_arms=4,read_bitwise_parity='failed in 8/8 CPU prefix comparisons',production_changed=False,profile_source_binding='archived measured source; current CLI removes unused --steps option')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
