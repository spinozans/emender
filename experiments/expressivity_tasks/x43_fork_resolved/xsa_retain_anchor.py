"""Replay registered probe unchanged, retaining its existing final CPU snapshot.

Use within a broker-leased detached wrapper. Never accept a retained checkpoint
without the matching anchor receipt. Checkpoints live outside the repository.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from experiments.expressivity_tasks.x45_matrix_emender import protocol_correction as pc
from experiments.expressivity_tasks.x45_matrix_emender import k15_probe_composition as k15


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('base',choices=['gdn2','k15']);ap.add_argument('directory',type=Path);a=ap.parse_args()
    if not os.environ.get('GPU_LEASE_HELD') or os.environ['GPU_LEASE_HELD']=='7':raise RuntimeError('lease required')
    m,arm_args,expected,engine_sha = ((pc,['--label','gdn2_speedrun_rescored'],1.3521734476089478,'2610942b448416d7365474130f1dc3e17056295b0defbe775b8128518a53b8b9') if a.base=='gdn2' else (k15,['--arm','composition'],1.3811104467936925,'e769a26157ca3da36569258289476f0c1e74e9b2e33e3d187e8b185076349d2c'))
    if hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()!=engine_sha: raise ValueError('engine hash')
    root=a.directory.resolve()/a.base;root.mkdir(exist_ok=False,parents=True); final=root/'final.pt'; out=root/'probe'; train=m._train_corrected
    def retain(*args,**kwargs):
        run=train(*args,**kwargs)
        if run['step']!=204 or 'final' not in run['snapshots']: raise RuntimeError('missing registered endpoint')
        tmp=final.with_suffix('.partial');torch.save(run['snapshots']['final'],tmp);os.replace(tmp,final);return run
    m._train_corrected=retain
    sys.argv=[m.__file__,'arm',*arm_args,'--seed','3401','--output',str(out)];m.main()
    rows=[json.loads(line) for line in (out/'scores.jsonl').read_text().splitlines()]
    if [r['eval_nll'] for r in rows if r['kind']=='post_cooldown']!=[expected]:raise ValueError('anchor mismatch: do not use snapshot')
    with final.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    (root/'anchor.json').write_text(json.dumps(dict(base=a.base,seed=3401,anchor=expected,state=str(final),sha256=sha,engine_sha256=engine_sha),indent=2)+'\n')

if __name__=='__main__':main()
