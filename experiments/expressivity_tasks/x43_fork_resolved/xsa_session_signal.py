"""Run PREREGISTRATION_SESSION_SIGNAL.md (CPU only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from experiments.expressivity_tasks.x42_pile_scale.lattice import Stream, flagship_identity


def document_scores(exe, piece):
    names = ['off', 'short', 'full', 'rotated']
    losses = dict.fromkeys(names, 0.0)
    correct = dict.fromkeys(names, 0)
    count = extra = short_hit = full_hit = 0
    with subprocess.Popen([exe, 'session', '8'], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        def rpc(cmd):
            proc.stdin.write(cmd.encode()+b'\n');proc.stdin.flush()
            return json.loads(proc.stdout.readline())
        rpc('append '+piece[:256].hex())
        for t in range(256, 2048):
            if t % 512 == 0: rpc('seal')
            local = piece[t-256:t]; pat = local[-8:]
            row = rpc('branch '+pat.hex()); full = np.array(row['counts'])
            if row['end']: raise ValueError('unexpected END')
            short = np.zeros(256, dtype=np.int64)
            j = local.find(pat)
            while j >= 0 and j+8<len(local):
                short[local[j+8]] += 1; j=local.find(pat,j+1)
            if (short>full).any(): raise ValueError('short not subset of full')
            q = (np.bincount(np.frombuffer(local, dtype=np.uint8), minlength=256)+1/256)/257
            y = piece[t]
            for name, evidence in [('off', 0), ('short', short), ('full', full), ('rotated', np.roll(full,1))]:
                v = q + evidence
                losses[name] -= math.log(v[y]/v.sum()); correct[name] += int(v.argmax()==y)
            extra += int(full.sum()>short.sum()); full_hit += int(full.any()); short_hit += int(short.any()); count+=1
            if rpc(f'append {y:02x}') != {'visible':t+1}: raise ValueError('append offset')
        proc.stdin.close()
        if proc.wait(timeout=10): raise RuntimeError('unclean server exit')
    return dict(targets=count, nll={k:v/count for k,v in losses.items()}, correct=correct,
                extra_positions=extra, full_supported=full_hit, short_supported=short_hit)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('exe', type=Path); ap.add_argument('out',type=Path);a=ap.parse_args()
    ident=flagship_identity(); stream=Stream(ident['source'],ident['manifest']); rows=[];seen=set()
    with stream.source.open('rb') as raw:
        for d in stream.documents('calibration'):
            if d['length']<2049 or d['doc'] in seen: continue
            raw.seek(d['offset']);text=raw.read(d['length'])
            if hashlib.sha256(text).hexdigest()!=d['sha256']: raise ValueError('document hash mismatch')
            rows.append(dict(document=d,**document_scores(a.exe.resolve(),text[:2049])));seen.add(d['doc'])
            if len(rows)==32:break
    if len(rows)!=32:raise ValueError('not enough documents')
    n=sum(r['targets'] for r in rows);scores={k:sum(r['nll'][k]*r['targets'] for r in rows)/n for k in rows[0]['nll']}
    summary=dict(targets=n,nll=scores,short_minus_full=scores['short']-scores['full'],
                 full_better_documents=sum(r['nll']['short']>r['nll']['full'] for r in rows),
                 byte_accuracy={k:sum(r['correct'][k] for r in rows)/n for k in scores},
                 support={k:sum(r[k] for r in rows)/n for k in ['extra_positions','full_supported','short_supported']})
    a.out.write_text(json.dumps(dict(identity=ident,binary_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),summary=summary,rows=rows),indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
