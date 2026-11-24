"""CPU-only aggregation of retained J per-document training and scoring receipts."""
import json
from pathlib import Path
import sys
from statistics import mean, median
from .k15_coupled_summary import interval, records


def work(rows):
    return {k:sum(r['work'][k] for r in rows) for k in rows[0]['work']}


def summarize(out):
    train=records(out/'console'); terminal=train.pop(); assert terminal['complete'] and len(train)==256
    result=dict(terminal=terminal,train=dict(docs=len(train),active=sum(r['active'] for r in train),nonzero=sum(r['grad']>0 for r in train),work=work(train),mean_kl=mean(r['kl'] for r in train if r['kl'] is not None)))
    for name in ('cal','confirmation'):
        if not (out/(name+'.jsonl')).exists():continue
        rows=records(out/(name+'.jsonl')); assert len(rows)==48
        panel=dict(work_per_full_policy_panel_pass=work(rows),immediate=sum(r['immediate'] for r in rows),active=sum(r['active'] for r in rows),ms={},loss={},contrasts=json.loads((out/(name+'_gate.json')).read_text())); result[name]=panel
        for mode in rows[0]['ms']:
            panel['ms'][mode]=median(r['ms'][mode] for r in rows)
            panel['loss'][mode]=[mean(r['modes'][mode][k] for r in rows if r['modes'][mode][k] is not None) for k in range(3)]
        panel['fixed_gain']=[interval([r['modes']['off'][k]-r['modes']['fixed'][k] for r in rows if r['modes']['off'][k] is not None]) for k in range(3)]
    return result

if __name__=='__main__': print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
