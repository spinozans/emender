"""Freeze neural comparison targets, then materialize independent-oracle-checked rows.

NPZ record: increasing t=256..2047 per document. Sparse (row, byte, count)
triples give full and bounded histograms. Store no raw target text in git.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from experiments.expressivity_tasks.x42_pile_scale.lattice import Stream, flagship_identity

HERE = Path(__file__).resolve().parent
BINARY = 'fdf5f97cb23d35e427161b60f9d7be47f5eeed6fd0bbf1a4bf4b81dbb6c0f1f0'


def bind(stream, split, limit):
    docs=[]; ids=set(); targets=hashlib.sha256(); pieces=[]
    with stream.source.open('rb') as f:
        for d in stream.documents(split):
            if d['doc'] in ids or d['length']<2049: continue
            f.seek(d['offset']); raw=f.read(d['length'])
            if hashlib.sha256(raw).hexdigest()!=d['sha256']: raise ValueError(d)
            piece=raw[:2049]; pieces.append(piece); targets.update(piece[256:2048]); docs.append(d); ids.add(d['doc'])
            if len(docs)==limit:break
    if len(docs)!=limit:raise ValueError('missing docs')
    return dict(split=split, documents=docs, target_sha256=targets.hexdigest(), targets=limit*1792), pieces


def generate(exe, pieces, path):
    full=[];short=[];row=0
    for raw in pieces:
        with subprocess.Popen([exe,'session','8'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as p:
            def rpc(cmd):
                p.stdin.write(cmd.encode()+b'\n');p.stdin.flush();return json.loads(p.stdout.readline())
            rpc('append '+raw[:256].hex()); hist=defaultdict(Counter)
            for i in range(8,256): hist[raw[i-8:i]][raw[i]]+=1
            for t in range(256,2048):
                if t%512==0:rpc('seal')
                key=raw[t-8:t]; answer=rpc('branch '+key.hex()); freq=hist[key]
                if answer!={'counts':[freq.get(b,0) for b in range(256)],'end':0}:raise ValueError('oracle mismatch')
                full.extend((row,b,c) for b,c in sorted(freq.items()))
                freq_s=Counter(); j=raw.find(key,t-256,t)
                while j>=0 and j+8<t:
                    freq_s[raw[j+8]]+=1;j=raw.find(key,j+1,t)
                short.extend((row,b,c) for b,c in sorted(freq_s.items()));row+=1
                hist[key][raw[t]]+=1
                if rpc('append '+raw[t:t+1].hex())!={'visible':t+1}: raise ValueError('cursor mismatch')
            p.stdin.close()
            if p.wait(timeout=10):raise RuntimeError('server exit')
    np.savez_compressed(path,full=np.array(full,dtype=np.int32).reshape(-1,3),short=np.array(short,dtype=np.int32).reshape(-1,3))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),audited_rows=row)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('exe',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    if hashlib.sha256(a.exe.read_bytes()).hexdigest()!=BINARY:raise ValueError('unregistered binary')
    a.out.mkdir(exist_ok=False,parents=True);identity=flagship_identity();stream=Stream(identity['source'],identity['manifest'])
    cal,calbytes=bind(stream,'calibration',32);test,testbytes=bind(stream,'final_test',64)
    pilot=json.loads((HERE/'receipts/xsa_session_signal.json').read_text())
    if cal['documents']!=[r['document'] for r in pilot['rows']]:raise ValueError('calibration drift')
    for k in ('doc','sha256'):
        if {d[k] for d in cal['documents']} & {d[k] for d in test['documents']}:raise ValueError('overlap')
    manifest=dict(identity=identity,calibration=cal,final_test=test,binary=BINARY)
    (a.out/'targets.json').write_text(json.dumps(manifest,indent=2)+'\n'); evidence={}
    for split,pieces in [('calibration',calbytes),('final_test',testbytes)]:
        evidence[split]=generate(a.exe.resolve(),pieces,a.out/(split+'.npz'))
    evidence['targets_manifest_sha256']=hashlib.sha256((a.out/'targets.json').read_bytes()).hexdigest()
    (a.out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n'); print(json.dumps(evidence))

if __name__=='__main__':main()
