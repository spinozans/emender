"""Summarize retained D receipts (CPU only; no intervention on training)."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

import numpy as np


def measure(out):
    train = [json.loads(line) for line in (out/'train.jsonl').open()]
    held = [json.loads(line) for line in (out/'eval.jsonl').open()]
    docs = {r['doc'] for r in train}; test = {r['doc'] for r in held}
    assert len({(r['doc'],r['start']) for r in train}) == len(train) == 4096
    assert not docs.intersection(test) and len(test) == len(held) == 64
    nll = defaultdict(list); latency = defaultdict(list); strata = defaultdict(list)
    work = defaultdict(Counter); choices = Counter(); lives=[]; endings=Counter(); changed=0
    for row in held:
        raw=bytes.fromhex(row['raw']); modes=row['modes']; true=modes['none']['trace']; birth=[None,None]
        for mode,data in modes.items():
            nll[mode].append(np.mean(data['loss'])); latency[mode].append(data['ms'])
            if mode in ('off','fixed'):
                work[mode]['read_RPC']+=7 if mode=='fixed' else 0
                continue
            for ev in data['trace']:
                work[mode]['read_RPC']+=len(ev['calls'])
                for _,reply in ev['calls']:
                    work[mode]['tail_starts']+=reply.get('tail_starts',0)
                    work[mode]['hits']+=len(reply.get('positions',[]))
        for ev,other in zip(true,modes['address']['trace']):
            changed+=ev.get('saved_pos')!=other.get('saved_pos')
        newest_support='none'
        for j in range(128):
            if j and j%16==0:
                ev=true[j//16-1];s,op=ev['slot'],ev['op'];choices[s,op]+=1
                if op!=2:
                    if birth[s] is not None:
                        lives.append(j-birth[s]); endings['retire' if op==3 else 'replace']+=1
                    birth[s]=None if op==3 else j
                hits=[r['positions'] for _,r in ev['calls'] if 'positions' in r]
                newest_support=('many' if len(hits[0])>1 else 'one') if hits else 'none'
            ages=[j-b for b in birth if b is not None];age=max(ages,default=-1)
            domain='ascii_prose_proxy' if sum(65<=b<=90 or 97<=b<=122 or b==32 for b in raw)>=96 else 'other_text_proxy'
            labels=[domain,'support_'+newest_support,'age_'+('none' if age<0 else '<16' if age<16 else '16-31' if age<32 else '32+'), 'boundary_delay_'+str((j%16)//4*4)]
            for label in labels:
                strata[label].append([modes[m]['loss'][j]-modes['none']['loss'][j] for m in ('clear','address','payload','fixed')])
        for b in birth:
            if b is not None: lives.append(128-b); endings['window_end']+=1
    bs=np.random.default_rng(3407).integers(0,len(held),(10000,len(held)))
    return dict(train_documents=len(docs),train_actions=dict(Counter(str(a) for r in train for a in r['actions'])),
                nll={m:np.mean(v) for m,v in nll.items()}, control_minus_true_CI={m:np.quantile((np.array(v)-nll['none'])[bs].mean(1),[.025,.975]).tolist() for m,v in nll.items()},
                window_ms_p50_p95={m:np.quantile(v,[.5,.95]).tolist() for m,v in latency.items()},work=dict(work),
                choices={str(k):v for k,v in choices.items()},address_changes=changed, lifetime_hist=dict(Counter(lives)),endings=dict(endings),
                strata_clear_address_payload_fixed_minus_true={k:[len(v),*np.mean(v,axis=0).tolist()] for k,v in strata.items()})

if __name__=='__main__': print(json.dumps(measure(Path(sys.argv[1])),indent=2))
