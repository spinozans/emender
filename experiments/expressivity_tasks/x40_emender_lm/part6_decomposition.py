"""Separate replay from core backward using actual launch correlations."""
import argparse
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,default=ROOT/'receipts/part5/pilot_B128_bounded');p.add_argument('--output',type=Path,default=ROOT/'receipts/part6/historical_backward_decomposition.json');a=p.parse_args();a.pilot=a.pilot.resolve()
    events=json.loads(gzip.decompress((a.pilot/'trace.json.gz').read_bytes()))['traceEvents']
    runtime=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')]
    device=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    def correlations(name):
        ranges=[e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name')==name]
        ids={e.get('args',{}).get('correlation') for e in runtime if any((name=='x40::backward' or e.get('tid')==r.get('tid')) and r['ts']<=e['ts']<=r['ts']+r['dur'] for r in ranges)}
        ids.discard(None);return ids,ranges
    back,back_ranges=correlations('x40::backward');replay,replay_ranges=correlations('x36::replay_backward')
    native_forward,_=correlations('x36::scan_forward')
    recompute=back & native_forward - replay
    def duration(ids):return sum(e.get('dur',0) for e in device if e.get('args',{}).get('correlation') in ids)/1000
    replay_ms=duration(replay);recompute_ms=duration(recompute);core_ms=duration(back-replay-recompute);total=replay_ms+core_ms+recompute_ms
    wall=json.loads((a.pilot/'wall.json').read_text());mean=sum(wall['seconds'])/len(wall['seconds']);tokens=wall['batch']*wall['context'];budget=tokens/100000
    result=dict(source=str(a.pilot.relative_to(ROOT)),scope='Separately profiled update. CPU user_annotation ranges only: backward includes autograd worker launches temporally within its synchronous host range; replay uses its worker thread. GPU mirrored annotation ranges are excluded. Summed GPU durations are not an additive wall identity.',
                backward_device_ms=total,replay_device_ms=replay_ms,core_backward_device_ms=core_ms,recomputed_native_forward_device_ms=recompute_ms,
                replay_fraction_of_backward_device_busy=replay_ms/total,core_fraction_of_backward_device_busy=core_ms/total,recompute_fraction_of_backward_device_busy=recompute_ms/total,
                backward_cpu_range_ms=sum(e['dur'] for e in back_ranges)/1000,
                replay_cpu_ranges_ms=sum(e['dur'] for e in replay_ranges)/1000,
                sustained_mean_seconds=mean,target_seconds=budget,required_savings_seconds=mean-budget,
                hypothetical_zero_replay_wall_seconds=mean-replay_ms/1000,
                hypothetical_zero_replay_scope='Optimistic subtraction for arithmetic only; GPU busy is not wall attribution and this is not an achieved timing or a floor.',
                irreducible_floor_established=False)
    result['replay_launch_resources']=[dict(duration_us=e.get('dur',0),**e.get('args',{})) for e in device if e.get('name')=='reverse_kernel']
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
