"""Attended, bounded checkpoint occupancy samples; never controls the trainer."""
import json
from pathlib import Path


def main():
    out=Path(__file__).with_name('receipts')/'c14_b_v2'
    path=out/'training.json';samples=out/'occupancy_time_samples.jsonl'
    if path.exists():
        t=json.loads(path.read_text());previous=0
        if samples.exists():previous=json.loads(samples.read_text().splitlines()[-1])['step']
        if t['steps']>previous:
            row=dict(step=t['steps'],active_seconds=t['active_seconds'],last_window_index=t['next_window_index']-1,
                checkpoint_sha256=t['checkpoint_sha256'],physical_gpu=t['config']['physical_gpu'],
                heads={k:{name:v[name] for name in ('final_array','final_stream','peak_occupancy','peak_array','peak_stream','budget','array_read_hit_rate')}|{
                    'array_read_opportunities':v['counts']['array_read_opportunities'],
                    'array_slots_found':v['counts']['array_slots_found'],'array_slots_read':v['counts']['array_slots_read']} for k,v in t['telemetry'].items()},
                scope='Last independent row at each completed observed checkpoint; peaks/findability counts are cumulative; no cross-window continuation')
            with samples.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    console=out/'train_console.txt'
    with console.open('rb') as f:
        f.seek(max(0,console.stat().st_size-16384));lines=f.read().decode().splitlines()
    if lines:
        last=json.loads(lines[-1]);print(json.dumps({k:last[k] for k in ('step','active_seconds','targets','nll','status','steps','actual_bytes_targets') if k in last}))
if __name__=='__main__':main()
