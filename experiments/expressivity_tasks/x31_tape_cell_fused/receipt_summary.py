"""Summarize measured gates and label derived checkpoint storage bounds."""
import json
from pathlib import Path
from .record_parity import source_hashes

ROOT=Path(__file__).resolve().parent


def main():
    def read(name):return json.loads((ROOT/name).read_text())
    parity=read('parity_part5.json')
    if not parity['full_parity'] or parity['source_sha256']!=source_hashes():
        raise RuntimeError('current source must have a passing parity attestation')
    mqar=read('mqar_part5.json');ladder=read('ladder_part5.json')
    cost=read('cost_part5.json');append=read('append_cost_part5.json')
    checkpoint=next(x for x in cost['measurements'] if x['engine']=='x31_checkpoint')
    base=next(x for x in cost['measurements'] if x['engine']=='x31' and x['slots']==11776 and x['heads']==4)
    if checkpoint['num_warps']!=base['num_warps']:
        raise RuntimeError('checkpoint and base launch geometries must match')
    n,t,dk,dv=256,16,64,128
    bounds=[]
    for s in (736,11776):
        capacity=(6*t+8)*s+t*(dk+dv+64)+(s+3*t)*(16*s.bit_length()+32)
        bounds.append(dict(slots=s,derived_reserved_bytes=n*(20*capacity+16*t+8),
                           derived_dense_record_lower_bound=n*(5*t+4)*s))
    assert bounds[-1]['derived_reserved_bytes']==checkpoint['journal_reserved_bytes']
    assert bounds[-1]['derived_dense_record_lower_bound']<=checkpoint['journal_recorded_entries']
    final=ladder['validation'][-1]
    best=max(ladder['validation'],key=lambda x:x['report']['balanced_accuracy'])
    result=dict(parity_tests=parity['tests'],full_parity=True,
        mqar=dict(updates=mqar['updates'],mean_per_class_recall=mqar['best'],gate=mqar['gate']),
        ladder=dict(updates=ladder['updates'],final=final,best=best,gate=ladder['gate']),
        forward_scope=cost['scope'],timing_gates=cost['gates'],
        x30_speedups=cost['x30_speedups'],heads_ratios=cost['heads16_to_1_by_slots'],
        checkpoint=dict(observed_fractional_time_delta=cost['checkpoint_overhead'],
            timing_gate=cost['gates']['checkpoint_overhead_le_5pct'],
            base_state_bytes=checkpoint['base_state_bytes'],
            reserved_bytes=checkpoint['journal_reserved_bytes'],
            recorded_entries=checkpoint['journal_recorded_entries'],
            used_record_bytes=20*checkpoint['journal_recorded_entries'],
            private_forward_replay_copy_bytes=checkpoint['private_replay_copy_bytes'],
            budget_scaling=bounds,storage_independence_gate=False,
            note='Dense physical vector spills and conservative capacity scale with S; not compact logical-event storage.'),
        append_phase_gates={str(x['slots']):dict(speedup=x['speedup'],gate=x['gate']) for x in append['measurements']},
        collection_errors=cost['errors'],all_registered_targets_passed=False)
    (ROOT/'summary_part5.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
