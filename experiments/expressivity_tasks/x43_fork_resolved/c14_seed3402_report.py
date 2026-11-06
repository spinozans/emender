"""Fail-closed paired sign harvest after both frozen evaluations complete."""
import json
from pathlib import Path
from .c14_documents import sha
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c14_b_seed3402'

def main():
    arms={}
    for arm in ('budgeted','unbudgeted'):
        o=OUT/arm;t=json.loads((o/'training.json').read_text());e=json.loads((o/'eval_resume.json').read_text())
        assert t['status']=='completed_matched_exposure' and (t['steps'],t['actual_bytes_targets'])==(35419,71677042)
        assert e['status']=='complete' and (e['windows'],e['targets'],e['next_shard'])==(3431242,434204084,26807)
        assert len(e['sums'])==66
        table={k:dict(sum=v[0],targets=v[1],nll=v[0]/v[1]) for k,v in e['sums'].items()}
        assert table['B1_on/final_test/total']['targets']==214097022
        assert sha(Path(t['checkpoint']))==t['checkpoint_sha256']
        arms[arm]=dict(training={k:t[k] for k in ('steps','actual_bytes_targets','active_seconds','checkpoint','checkpoint_sha256')},table=table,training_receipt_sha256=sha(o/'training.json'),eval_receipt_sha256=sha(o/'eval_resume.json'))
        if arm=='unbudgeted':
            for v in t['telemetry'].values():assert v['peak_occupancy']<257 and v['counts']['capacity_merges']==0
    b,u=arms['budgeted'],arms['unbudgeted'];assert b['table'].keys()==u['table'].keys()
    for k in b['table']:assert b['table'][k]['targets']==u['table'][k]['targets']
    delta=u['table']['B1_on/final_test/total']['nll']-b['table']['B1_on/final_test/total']['nll']
    sign='near-zero descriptive threshold' if abs(delta)<.005 else 'holds negative sign' if delta<0 else 'flips positive sign'
    v=dict(status='complete',seed=3402,delta=delta,verdict=sign,arms=arms,prior_seed3401_delta=-.01942916080979984,limits='Two seeds; descriptive .005 practical threshold is not statistical uncertainty. Capacity/implementation/trajectory remain confounded.')
    (OUT/'sign_verdict.json').write_text(json.dumps(v,indent=2)+'\n')
    lines=['# C14-B second-seed twin — the sign verdict','',f"Seed3402: unbudgeted minus budgeted final-test NLL = **{delta:+.9f} nats/byte**: **{sign}**. Prior seed3401 was-.019429161.",'',f"Budgeted {b['table']['B1_on/final_test/total']['nll']:.9f}; unbudgeted {u['table']['B1_on/final_test/total']['nll']:.9f}. Both35419 updates /71677042 masked raw targets. Both evals3431242 windows /434204084 targets; final-test214097022 per mode. All66 row keys and counts match. Full sums and hashes in receipts/c14_b_seed3402/sign_verdict.json.",'', 'Budgeted retains historical32/64 kernel; unbudgeted uses source-checked certified phase2 rowkeys, with phase1–3 equivalence citations in registration.json. Independent document windows, optimizer, precision and dropout unchanged. Fresh seed is the registered intervention; matched stopping applies to both arms. No frozen model or corpus was changed. No checkpoint committed.','', 'Two seeds do not establish a general sign law. The practical near-zero threshold is descriptive, not a significance/noise test. Capacity, implementation, trajectory and training compute remain confounded. No resilient DiLoCo work; gap IDs inapplicable.','', '| row | targets | budgeted NLL | unbudgeted NLL |','|---|---:|---:|---:|']
    for k in b['table']:lines.append(f"|{k}|{b['table'][k]['targets']}|{b['table'][k]['nll']:.9f}|{u['table'][k]['nll']:.9f}|")
    (ROOT/'RESULT_C14_B_SEED3402.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(delta=delta,verdict=sign)))

if __name__=='__main__':main()
