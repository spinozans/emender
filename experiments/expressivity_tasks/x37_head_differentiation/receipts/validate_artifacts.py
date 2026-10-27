"""CPU-only integrity checks over the completed experiment's raw evidence."""
import hashlib
import json
import math
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parent
ROOT=R.parent

def read(name):
    return json.loads((R/name).read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest=read('gate_sources.json')
with tarfile.open(R/'gated_sources.tar.gz') as archive:
    assert set(archive.getnames())==set(manifest)
    for name,expected in manifest.items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
assert digest(R/'gated_sources.tar.gz')==read('source_archive.json')['sha256']
assert digest(R/'resume_pinned.py')==read('snapshot_validation.json')['loader_sha256']
for filename,count in (('oracle.xml',141),('mechanism.xml',4),('pinned_oracle.xml',113)):
    cases=ET.parse(R/filename).getroot().findall('.//testcase')
    assert len(cases)==count,(filename,len(cases))
    assert all(c.find(tag) is None for c in cases for tag in ('failure','error','skipped'))
current={name:digest(ROOT.parent/name) for name in manifest}
assert all(current[name]==expected for name,expected in manifest.items()
           if name.startswith('x37_head_differentiation/'))
(R/'source_drift_final.json').write_text(json.dumps(dict(
    imported_source_authority='gate_sources.json + byte-verified gated_sources.tar.gz',
    own_gated_sources_unchanged=True,
    upstream_working_tree_differences={name:dict(gated=expected,current=current[name])
        for name,expected in manifest.items() if expected!=current[name]}),indent=2)+'\n')
summary=read('summary.json')
assert summary['status']=='completed'
assert summary['control_tokens']==100003840 and summary['arm_tokens']==100007936
assert summary['endpoint_token_difference']==4096
assert read('control.json')['primary']['identical_stories']==16
assert all(not d['bit_equal'] for distances in read('control_checkpoint_audit.json')['head_projection_distances'].values() for d in distances.values())
common=summary['arms']['b']['training']['fingerprint']
control_run=json.loads((ROOT.parent/'x34_lm_pilot/receipts/pilot_tape_lm.json').read_text())
assert control_run['fingerprint']['tokenizer']==common['tokenizer']
assert control_run['fingerprint']['data']==common['data_receipt']
assert control_run['fingerprint']['seed']==common['seed']==3401
assert control_run['fingerprint']['batch']==8
assert common['batch']==64
for arm in ('b','c','d'):
    fp=summary['arms'][arm]['training']['fingerprint']
    assert all(fp[key]==common[key] for key in
               ('tokenizer','data_receipt','train_bin','valid_bin','seed','batch','context'))
checked_rows=0
for arm in ('b','c','d'):
    training=read(f'train_{arm}.json')
    assert training['status']=='completed' and training['steps']==3052
    assert training['tokens']==100007936
    assert training['fingerprint']['sources']==manifest
    assert training['physical_gpu'] in ('0','1','5','6')
    assert training['fingerprint']['batch']==64 and training['fingerprint']['context']==512
    rows=[json.loads(line) for line in (ROOT/'runs'/arm/'curve.jsonl').read_text().splitlines()]
    assert len(rows)==3052
    assert [r['step'] for r in rows]==list(range(1,3053))
    assert all(r['tokens']==r['step']*32768 and r['replay_node']=='ReplayBackward' for r in rows)
    assert all(math.isfinite(r[name]) for r in rows for name in ('loss','grad_norm','gpu_step_ms'))
    assert all(r['gpu_step_ms']>0 for r in rows)
    checked_rows+=len(rows)
    params=read(f'parameters_{arm}.json')
    assert params['relative_total_difference']<.001
    assert abs(params['arm']['total']/params['base_lm']['total']-1)==params['relative_total_difference']
    for step in (0,64,512,1536,3052):
        evaluation=read(f'eval_{arm}_{step}.json')
        assert evaluation['tokens']==step*32768
        stories=evaluation['stories']
        assert len(stories)==16
        peaks=[s['telemetry'][0]['peak_live_per_head'] for s in stories]
        assert all(len(p)==4 for p in peaks)
        assert sum(len(set(p))==1 for p in peaks)==evaluation['primary']['identical_stories']
        for story in stories:
            telemetry=story['telemetry'][0]
            assert len(telemetry['routing_per_head'])==4
            assert all(sum(head.values())==story['target_tokens'] for head in telemetry['routing_per_head'])
        if arm in ('c','d') and step>0:
            win=evaluation['W_in']
            assert len(win['distance_from_init'])==4
            assert all(math.isfinite(v) and v>0 for v in win['distance_from_init'])
            assert all(math.isfinite(v) and v>0 for v in win['pairwise_distance'].values())
            assert all(math.isfinite(v) and v>0 for v in win['gradient_norm'])
    last=read(f'eval_{arm}_3052.json')
    assert summary['arms'][arm]['primary']==last['primary']
    assert summary['arms'][arm]['validation']==last['validation']
probe=read('winner_probe.json')
assert probe['arm']==summary['winner']
assert len(probe['stories'])==16
assert [r['k'] for r in probe['depth']]==[16,32,64,128,256,512,1152]
assert [r['context'] for r in probe['context']]==[512,1024,2048,4096]
assert all(r['windows']==2 for r in probe['context'])
result=dict(passed=True,training_rows_checked=checked_rows,arms=3,tokens_per_arm=100007936,
            evaluations_checked=15,stories_checked=240,source_files_verified=len(manifest),
            oracle_tests=141,mechanism_tests=4,pinned_oracle_and_mechanism_tests=113,
            winner=summary['winner'],scope='Raw evidence integrity; prediction failures remain results, not waived checks.')
(R/'artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
