"""Replay rate arithmetic and bound disk scenarios from immutable snapshots."""
import datetime as dt
import json
from pathlib import Path

P=Path(__file__).resolve().parent

def read(name): return json.loads((P/name).read_text())
def rows(name): return [json.loads(s) for s in (P/name).read_text().splitlines()]
def segment(rs):
    starts=[i for i,r in enumerate(rs) if i==0 or r['seconds']<rs[i-1]['seconds'] or r['windows']<=rs[i-1]['windows']]
    return rs[starts[-1]:]
def interval(a,b):
    sec=b['seconds']-a['seconds']; n=b['positions']-a['positions']
    return dict(start_window=a['windows'],end_window=b['windows'],start_seconds=a['seconds'],end_seconds=b['seconds'],seconds=sec,targets=n,targets_per_second=n/sec)
def utc(t):return dt.datetime.fromtimestamp(t,dt.timezone.utc).isoformat()

before=segment(rows('before_console.jsonl')); after=segment(rows('after_console.jsonl'))
service=rows('after_service_journal.jsonl'); phase=next(r for r in reversed(service) if r['event']=='phase_start')
journal=rows('after_extraction_journal.jsonl'); start=next(r for r in reversed(journal) if r['event']=='phase_start')
parity=read('parity.json'); snap=read('after_snapshot.json')
assert after[-1]['windows']>before[-1]['windows'], 'journal did not advance'
assert after[0]==before[0], 'invocation changed'
pre=interval(before[0],before[-1])
# Exclude the verifier's own native-server lifetime plus one full completed shard.
post=[r for r in after if phase['time']+r['seconds']>parity['verifier_end']+5]
clean=[dict(label='pre-oracle uninterrupted endpoint interval',**pre)]
if len(post)>1:clean.append(dict(label='post-oracle uninterrupted endpoint interval',**interval(post[0],post[-1])))
longest=max(clean,key=lambda x:x['seconds'])
blocks=[]
for i in range(0,len(after)-16,16):
    blocks.append(interval(after[i],after[i+16]))
launch=read('../c14_union_service/launch.json')
launchrow=next(r for r in after if r['windows']==launch['windows'])
launch_start=dict(windows=start['windows'],positions=start['positions'],seconds=0)
short=interval(launch_start,launchrow); corrected=interval(after[0],launchrow)
baseline=launch['baseline_targets_per_second']
shards=[r for r in journal if r['event']=='shard_complete']
new=[r for r in shards if r['first_window_index']>=19968]
all_pos=sum(r['positions'] for r in shards); all_bytes=sum(r['bytes'] for r in shards)
new_pos=sum(r['positions'] for r in new); new_bytes=sum(r['bytes'] for r in new)
assert sum(r['windows'] for r in shards)>=after[-1]['windows']
union=1406579817; remaining=225880702-after[-1]['positions']
scenarios={}
for name,rate in [('longest_clean_observed',longest['targets_per_second']),('current_invocation_including_startup_and_spotcheck',(after[-1]['positions']-start['positions'])/after[-1]['seconds']),('accepted_baseline',baseline)]:
    scenarios[name]=dict(targets_per_second=rate,remaining_targets=remaining,remaining_hours=remaining/rate/3600,emender_boundary_utc=utc(snap['time']+remaining/rate))
write=dict(status='passed',observation_start_utc=utc(read('before_snapshot.json')['time']),observation_end_utc=utc(snap['time']),current_service_start_utc=utc(next(r for r in reversed(service) if r['event']=='service_start')['time']),snapshot_positions=after[-1]['positions'],snapshot_windows=after[-1]['windows'],accepted_baseline_targets_per_second=baseline,original_launch_slice=short,launch_slice_after_first_completed_shard=corrected,first_shard_startup_seconds=after[0]['seconds'],clean_intervals=clean,longest_clean_interval=longest,whole_invocation_endpoint_interval=interval(after[0],after[-1]),journal_blocks_2048_windows=blocks,rate_classification='The 4474/s launch slice includes startup; after its first completed shard the same slice exceeds baseline. Later journal blocks vary substantially under concurrent native index benchmarking. Observed sustained deficit is real for this interval, but does not establish a persistent implementation regression; attribution to contention/content is observational, not controlled.',projections=scenarios,disk=dict(free_bytes=snap['free_bytes'],floor_bytes=150000000000,all_committed_cache_bytes=all_bytes,all_committed_positions=all_pos,new_bytes_since_launch=new_bytes,new_positions_since_launch=new_pos,new_bytes_per_position=new_bytes/new_pos,new_tail_linear_union_bytes=all_bytes+(union-all_pos)*new_bytes/new_pos,prefix_mean_linear_union_bytes=union*all_bytes/all_pos,baseline_113_bytes_per_position_union_bytes=union*113,warning='159 GB is a content-dependent projection, not a proven upper bound. Prefix-mean and recent-tail scenarios both retained; the free-space floor is authoritative.'),no_rate_optimization=True,supervisor_channel='contact_supervisor unavailable in enabled tool catalog')
(P/'accounting.json').write_text(json.dumps(write,indent=2)+'\n')
print(json.dumps(write,indent=2))
