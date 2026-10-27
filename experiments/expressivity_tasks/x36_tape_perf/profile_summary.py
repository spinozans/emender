"""Attribute device activities by runtime correlation and CPU scan ranges."""
import json
from collections import Counter


def summarize(path):
    events=json.loads(path.read_text())['traceEvents']
    ranges={name:[e for e in events if e.get('ph')=='X' and e.get('name')==name]
            for name in ('x36::scan_forward','x36::replay_backward')}
    runtime=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')]
    activities=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    result={}
    for name,intervals in ranges.items():
        correlations={e.get('args',{}).get('correlation') for e in runtime
                      if any(e.get('tid')==r.get('tid') and r['ts']<=e['ts']<=r['ts']+r['dur']
                             for r in intervals)}
        correlations.discard(None)
        matched=[e for e in activities if e.get('args',{}).get('correlation') in correlations]
        result[name]={'activity_count':len(matched),'counts':dict(Counter(e['name'] for e in matched)),
                      'device_busy_ms':sum(e.get('dur',0) for e in matched)/1000}
    result['steady_sync_runtime']=[{'name':e['name'],'ts':e['ts'],'dur_us':e.get('dur',0)}
                                   for e in runtime if 'Synchronize' in e.get('name','')]
    result['dtoh']=[e['name'] for e in activities if 'dtoh' in e['name'].lower() or 'device to host' in e['name'].lower()]
    result['kernel_busy_ms_by_name']={name:sum(e.get('dur',0) for e in activities if e['name']==name)/1000
                                     for name in sorted({e['name'] for e in activities})}
    return result
