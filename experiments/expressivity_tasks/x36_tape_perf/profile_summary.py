"""Attribute device activities by runtime correlation and CPU scan ranges."""
import json
import gzip
from collections import Counter


def summarize(path):
    payload=path.read_bytes()
    if path.suffix=='.gz':payload=gzip.decompress(payload)
    events=json.loads(payload)['traceEvents']
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
        components={}
        for event in matched:
            kernel=event['name']
            component=(kernel if kernel in ('scan_kernel','reverse_kernel','rebase','initialize_free_words','materialize_state','setup_kernel')
                       else 'fills' if 'FillFunctor' in kernel or event.get('cat')=='gpu_memset'
                       else 'copies_transposes_mapping' if event.get('cat')=='gpu_memcpy' or 'copy' in kernel.lower()
                       else 'other_setup_or_reduction')
            components[component]=components.get(component,0)+event.get('dur',0)/1000
        result[name]['component_device_busy_ms']=components
    result['steady_sync_runtime']=[{'name':e['name'],'ts':e['ts'],'dur_us':e.get('dur',0)}
                                   for e in runtime if 'Synchronize' in e.get('name','')]
    result['dtoh']=[e['name'] for e in activities if 'dtoh' in e['name'].lower() or 'device to host' in e['name'].lower()]
    result['kernel_busy_ms_by_name']={name:sum(e.get('dur',0) for e in activities if e['name']==name)/1000
                                     for name in sorted({e['name'] for e in activities})}
    return result
