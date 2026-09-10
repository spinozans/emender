#!/usr/bin/env python3
"""Fail-closed aggregation for the bounded CUDA SR toy qualification."""
import argparse
import hashlib
import json
from pathlib import Path

from ndm.e97_atomic import publish_bytes_no_replace


def aggregate(root, world, qualifier_sha256):
    root=Path(root)
    results=[]
    for phase in ('produce','resume'):
        visible=None
        for rank in range(world):
            report=json.loads((root/f'{phase}-rank-{rank:02d}.json').read_text())
            assert report['schema']=='emender-sr-cuda-distributed-v1'
            assert report['status']=='passed' and report['phase']==phase
            assert report['rank']==rank and report['world_size']==world
            assert report['source_sha256']==qualifier_sha256
            route=report['device_identity']
            assert route['local_rank']==rank and route['current_cuda_device']==rank
            assert route['visible_devices'] and len(route['visible_devices'].split(','))==world
            if visible is None: visible=route['visible_devices']
            assert route['visible_devices']==visible
            assert [r['mode'] for r in report['reports']]==['ddp','diloco','hybrid']
            for r in report['reports']:
                mode=r['mode']
                assert r['status']=='passed'
                assert r['island_size']=={'ddp':world,'diloco':1,'hybrid':world//2}[mode]
                assert r['merge_count']==(0 if mode=='ddp' else 2)
                expected=json.loads((root/f'{mode}-rank-{rank:02d}-expected.json').read_text())
                assert r['final_state_sha256']==expected['final_state_sha256']
                assert hashlib.sha256((root/f'{mode}-rank-{rank:02d}.pt').read_bytes()).hexdigest()==expected['checkpoint_sha256']
            results.append(report)
    arithmetic=json.loads((root/'cuda-arithmetic-memory.json').read_text())
    assert arithmetic['status']=='passed'
    assert arithmetic['arithmetic']['counter_cpu_cuda_and_bucket_parity'] is True
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()}
    return {'schema':'emender-sr-cuda-qualification-summary-v1','status':'passed','world_size':world,
            'qualifier_sha256':qualifier_sha256,'rank_reports':len(results),
            'modes':['ddp','diloco','hybrid'],'cold_resume_exact':True,'files':files,
            'scope':'local small-model CUDA/NCCL and synthetic arithmetic only; no full E97 backward, trainer/loader integration, ROCm, Frontier, or elastic conformance'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--qualifier-sha256',required=True)
    a=p.parse_args(); result=aggregate(a.root,a.world_size,a.qualifier_sha256)
    publish_bytes_no_replace(a.root/'summary.json',(json.dumps(result,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)
    print('SR_CUDA_QUALIFICATION_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':main()
