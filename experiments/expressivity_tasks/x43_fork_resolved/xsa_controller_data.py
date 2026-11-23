"""Bind and audit training evidence without changing the sealed final-test fixture."""
import argparse
import hashlib
import json
from pathlib import Path
from .xsa_neural_evidence import BINARY, HERE, Stream, flagship_identity, bind, generate

TEST = HERE / 'receipts/xsa_neural_evidence_v1'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_pieces(manifest, source):
    pieces=[]
    with Path(source).open('rb') as f:
        for d in manifest['documents']:
            f.seek(d['offset']); raw=f.read(d['length'])
            if hashlib.sha256(raw).hexdigest()!=d['sha256']: raise ValueError(d)
            pieces.append(raw[:2049])
    if hashlib.sha256(b''.join(x[256:2048] for x in pieces)).hexdigest()!=manifest['target_sha256']: raise ValueError('target digest')
    return pieces

def prepare(exe,out):
    if digest(exe)!=BINARY: raise ValueError('engine hash')
    out.mkdir(exist_ok=False,parents=True); identity=flagship_identity(); stream=Stream(identity['source'],identity['manifest'])
    train,pieces=bind(stream,'train',128); orig=json.loads((TEST/'targets.json').read_text()); test=orig['final_test']
    if orig['identity']!=identity: raise ValueError('source binding')
    for key in ('doc','sha256'):
        if {r[key] for r in train['documents']} & {r[key] for r in test['documents']}: raise ValueError('train/test overlap')
    evidence=generate(exe,pieces,out/'train.npz')
    manifest=dict(identity=identity,train=train,evidence=evidence,test_manifest_sha=digest(TEST/'targets.json'),engine=BINARY)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Audited train rows',evidence['audited_rows'],flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('exe',type=Path);a.add_argument('out',type=Path);args=a.parse_args();prepare(args.exe,args.out)
