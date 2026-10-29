"""Actual inclusive replay launch counts and recorded decision-density receipts."""
import gzip
import json
from pathlib import Path

R=Path(__file__).resolve().parent/'receipts/part6'


def profile(label):
    root=R/label
    if not (root/'trace.json.gz').exists():return dict(status='missing')
    wall=json.loads((root/'wall.json').read_text())
    events=json.loads(gzip.decompress((root/'trace.json.gz').read_bytes()))['traceEvents']
    ranges=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='x36::replay_backward' and e.get('ph')=='X']
    runtime=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')]
    device=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    rows=[]
    for interval in ranges:
        launches=[e for e in runtime if e.get('tid')==interval.get('tid') and interval['ts']<=e['ts']<=interval['ts']+interval['dur']]
        ids={e.get('args',{}).get('correlation') for e in launches};ids.discard(None)
        activities=[e for e in device if e.get('args',{}).get('correlation') in ids]
        rows.append(dict(cuda_launch_api_calls=sum('LaunchKernel' in e['name'] for e in launches),
                         kernels=sum(e['cat']=='kernel' for e in activities),
                         memcpy=sum(e['cat']=='gpu_memcpy' for e in activities),
                         memset=sum(e['cat']=='gpu_memset' for e in activities),
                         inclusive_gpu_activities=len(activities),device_ms=sum(e.get('dur',0) for e in activities)/1000))
    assert len(rows)==24
    return dict(status='passed',label=label,tokens_per_second=wall['tokens_per_second'],batch=wall['batch'],
                replay_calls=len(rows),per_replay=rows,mean={key:sum(r[key] for r in rows)/len(rows) for key in rows[0]},
                total={key:sum(r[key] for r in rows) for key in rows[0]})


def main():
    off=profile('pilot_B128_fusion_off_clear_registers0_chunk1024')
    on=profile('pilot_B128_fusion_on_clear_registers0_chunk1024')
    result=dict(off=off,on=on,scope='Separate profiled updates, complete replay CPU ranges on their worker thread. All correlated kernel/copy/set activities are inclusive; launch API counts are separately named. Sustained throughput comes from 20 unprofiled full training updates. Both comparison arms clear auxiliary graphs, use no checkpoint or journal packing, and retain the original stream/register policy.')
    if off['status']==on['status']=='passed':
        result.update(status='passed',removed_inclusive_activities_per_update=off['total']['inclusive_gpu_activities']-on['total']['inclusive_gpu_activities'],
                      removed_kernel_launches_per_update=off['total']['kernels']-on['total']['kernels'],
                      measured_speedup=on['tokens_per_second']/off['tokens_per_second'])
    else:result['status']='not-measured'
    sparse=R/'pilot_B128/decision_sparsity.json'
    if sparse.exists():
        rows=json.loads(sparse.read_text())['rows']
        result['decision_density']=dict(total_token_events=sum(r['total_token_events'] for r in rows),
            value_consuming_events=sum(r['value_consuming_events'] for r in rows),
            total_lanes=sum(r['lanes'] for r in rows),lanes_without_any_value_consumer=sum(r['lanes_without_any_value_consumer'] for r in rows),
            read_prefix_tokens_before_first_value_consumer=sum(r['read_prefix_tokens_before_first_value_consumer'] for r in rows),
            interpretation='The no-consumer lane and prefix counts describe eligibility only. They do not prove a wall-time saving or irreducible floor. Finer row-specific dependencies require a separate certified recurrence change.')
    (R/'fusion_comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
