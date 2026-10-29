"""Join observed OOM stacks, allocation arithmetic and isolated memory controls."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
R=ROOT/'receipts/part6'


def load(path):return json.loads(path.read_text()) if path.exists() else None


def memory_cell(path):
    d=load(path)
    if d is None:return dict(status='missing')
    return {key:d.get(key) for key in ('status','peak_allocated','peak_reserved','error','microbatch','accumulation_steps','checkpoint','legacy_checkpoint','packed','checkpoint_input_offload','allocator_expandable','measured_updates','warmups')}


def main():
    arithmetic=load(R/'memory_arithmetic.json')
    rows=[]
    for ledger in arithmetic['arms']:
        name=f"{ledger['scale']}_{ledger['arm']}_B{ledger['batch']}_T{ledger['context']}"
        historical=load(ROOT/'receipts/part5/surface'/f'{name}.json')
        allocator=load(R/'baseline'/f'{name}_allocator.json')
        before=load(R/'baseline/surface'/f'{name}.json') or historical
        row=dict(scale=ledger['scale'],arm=ledger['arm'],batch=ledger['batch'],context=ledger['context'],
                 historical_status=historical['status'],before_status=before['status'],before_error=before.get('error'),
                 static_ledger=ledger,after=memory_cell(R/'surface'/f'{name}.json'))
        if allocator:
            row['before_peak_allocated']=allocator.get('peak_allocated')
            row['before_peak_reserved']=allocator.get('peak_reserved')
            row['allocator_history']=allocator
            oom=[e for device in allocator['devices'] for e in device.get('oom_events',[])]
            if oom:row['dominant_term_at_oom']=oom[-1].get('dominant_logical_term')
            elif allocator['devices']:
                counts=allocator['devices'][0]['categories'];row['dominant_term_at_observed_high_water']=max(counts,key=counts.get) if counts else None
        if ledger['arm']=='pure' and historical['status']!='passed':
            row['two_update_controls']={variant:memory_cell(R/f'memory_{variant}'/f'{name}.json') for variant in ('legacy','recorded_physical','accumulated','packed','offloaded') if variant!='offloaded' or ledger['context']==2048 and ledger['scale']=='1.3B'}
        elif ledger['arm']=='transformer' and historical['status']!='passed':
            row['two_update_controls']={key:memory_cell(R/f'memory_transformer_{variant}'/f'{name}.json') for key,variant in (('recorded_physical','physical'),('accumulated','accumulated'))}
        rows.append(row)
    result=dict(scope='Before errors and measured allocator histories are observations, not estimates of required complete-run memory. Static full-stack sums are not concurrent peaks. Two-update controls include initialized Adam states on the second forward/backward and are not sustained throughput results.',
                ledger_scope='Allocation-stack families separate native stores, journals, replay scratch, weights and optimizer allocations. activations_and_other includes parameter gradients and framework workspaces; unattributed bytes remain explicit.',
                rows=rows,remaining_unattributed_terms='Activation-only and framework-gradient subcategories are not independently isolated; stack samples and complete byte accounting are retained.',
                transient_writer_reservation_retained=True,production_budget_policy=arithmetic['budget_policy'])
    (R/'memory_breakdown.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
