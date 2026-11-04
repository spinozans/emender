"""Compact attestations/archives for completed controls; never selects a model."""
import gzip,hashlib,json,sys
from pathlib import Path
from .c14_documents import sha
ROOT=Path(__file__).resolve().parent

def main():
    control,phase=sys.argv[1:3];assert control in ('append_only','unbudgeted') and phase in ('train','eval')
    base=ROOT/'receipts/c14_b_v2';out=ROOT/('receipts/c14_b_v2_'+control)
    t=json.loads((out/'training.json').read_text());assert t['status']=='completed_active_budget'
    assert t['active_seconds']>=2700
    for name,digest in t['config']['code_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(Path(t['checkpoint']))==t['checkpoint_sha256']
    def compact(rows):
        return {k:{n:v for n,v in r.items() if n!='sampled_occupancy'} for k,r in rows.items()}
    def cap(rows):
        if control=='unbudgeted':
            assert all(r['budget']==257 and r['peak_occupancy']<257 and r['counts']['capacity_merges']==0 for r in rows.values())
        else:assert all(r['counts']['stale_retirements']==0 for r in rows.values())
    cap(t['telemetry'])
    if phase=='train':
        result={k:t[k] for k in ('status','steps','actual_bytes_targets','active_seconds','next_window_index','checkpoint','checkpoint_sha256','config','keep_fraction','excluded_checkpoint_stalls')}
        result.update(control=control,steps_per_active_second=t['steps']/t['active_seconds'],telemetry=compact(t['telemetry']),
            raw_receipt_sha256=sha(out/'training.json'),finite_cap_never_bound=control=='unbudgeted')
        names=['training.json'];target='training_summary.json'
    else:
        v=json.loads((out/'verdicts.json').read_text());state=json.loads((out/'eval_resume.json').read_text())
        assert v['status']==state['status']=='complete'
        assert (state['windows'],state['targets'])==(3431242,434204084)
        assert v['PB1']['targets']==214097022
        cap(v['PB3']['evaluation_sampled'])
        b=json.loads((base/'verdicts.json').read_text())
        result={k:v[k] for k in ('status','PB1','PB2','PB4')}
        result.update(control=control,delta_vs_B1=v['PB1']['nll']-b['PB1']['nll'],
            training=json.loads((out/'training_summary.json').read_text()),
            PB3=dict(training=compact(v['PB3']['training']),evaluation_sampled=compact(v['PB3']['evaluation_sampled']),findability=v['PB3']['findability']),
            eval_config=state['config'],windows=state['windows'],targets=state['targets'],eval_seconds=state['elapsed_seconds'],
            raw_verdict_sha256=sha(out/'verdicts.json'),finite_cap_never_bound=control=='unbudgeted',
            caveat='One seed, registered equal active budgets; report different steps/target exposure. Conditional controls do not replace the selected B1 model or its PB verdicts.')
        names=['verdicts.json','eval_resume.json'];target='verdict_summary.json'
    for name in names:
        data=(out/name).read_bytes();packed=gzip.compress(data,mtime=0);assert gzip.decompress(packed)==data
        (out/(name+'.gz')).write_bytes(packed)
    (out/target).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(control=control,phase=phase,status=result['status'],steps=t['steps'],targets=t['actual_bytes_targets'],steps_per_second=t['steps']/t['active_seconds'],NLL=result.get('PB1',{}).get('nll'),delta_vs_B1=result.get('delta_vs_B1')),indent=2))
if __name__=='__main__':main()
