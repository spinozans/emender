"""Registered fixed-evidence neural comparison (offline teacher-forced scoring)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from .xsa_controller_data import TEST, digest, load_pieces
from .xsa_neural_evidence import BINARY, flagship_identity, Stream
from ..x45_matrix_emender import k15_probe_composition as k15, protocol_correction as pc
from ..x40_emender_lm.part6_surface import features
from ..x45_matrix_emender.matched_arms import build_gdn2
from ..x45_matrix_emender.velocity_ladder import _build_k10
from ..x45_matrix_emender.k15_composition import convert_model_to_composition
from ..x42_pile_scale.lattice import evaluate

GRID=(0.,1/16,1/4,1.,4.)
N=1792


def histogram(sparse, offset):
    rows,bs,counts=sparse.T
    keep=(rows>=offset)&(rows<offset+N)
    h=np.zeros((N,256));h[rows[keep]-offset,bs[keep]]=counts[keep];return h


def mix(qy,h,y,lam):
    return (qy+lam*h[np.arange(N),y])/(1+lam*h.sum(1))


@torch.no_grad()
def run(base,root,output):
    os.environ['X40_BATCHED_READS']='0';pc._require_lease();root=root/base
    anchor=json.loads((root/'anchor.json').read_text());state=root/'final.pt'
    module,expected=(pc,1.3521734476089478) if base=='gdn2' else (k15,1.3811104467936925)
    engine=('2610942b448416d7365474130f1dc3e17056295b0defbe775b8128518a53b8b9' if base=='gdn2' else 'e769a26157ca3da36569258289476f0c1e74e9b2e33e3d187e8b185076349d2c')
    with state.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    if not (anchor['base']==base and anchor['anchor']==expected and anchor['sha256']==sha and anchor['engine_sha256']==engine==digest(Path(module.__file__))):raise ValueError('anchor identity')
    if base=='gdn2':model,*_=build_gdn2(3401)
    else:
        model,*_=_build_k10(dict(k15._tuned_winner()),3401);model=convert_model_to_composition(model).cuda()
    model.load_state_dict(torch.load(state,map_location='cpu',weights_only=True));model.eval()
    identity=flagship_identity();stream=Stream(identity['source'],identity['manifest']);orig=evaluate(model,stream,'gdn2' if base=='gdn2' else 'matrix_ops','calibration')
    if orig['score']!=expected: raise ValueError(f'reloaded anchor mismatch: {orig}')
    manifest=json.loads((TEST/'targets.json').read_text()); refs=json.loads((TEST/'evidence.json').read_text())
    if manifest['binary']!=BINARY or manifest['identity']!=identity or refs['targets_manifest_sha256']!=digest(TEST/'targets.json'):raise ValueError('target binding')
    def probs(piece,length,y):
        ids=torch.tensor(list(piece[:2048]),device='cuda').reshape(-1,length); blocks=[]
        # Independent forwards; no cache or state between blocks.
        for x in ids.split(1):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if base=='gdn2': z=features(model.model,x,'gdn2',False)[0];head=model.model.head
                else: z=model.features(x);head=model.head
            logits=z.bfloat16().float() @ head.weight.bfloat16().float().T
            blocks.append(logits.double().softmax(-1).cpu().numpy())
        q=np.concatenate(blocks,axis=1)[0,255:2047];return q[np.arange(N),y]
    result=dict(base=base,anchor=anchor,original_eval=orig,manifest_sha=refs['targets_manifest_sha256']); keys=('off','on','short_hist','rotated','long_off','fixed_unigram')
    for split in ('calibration','final_test'):
        p=TEST/(split+'.npz');
        if digest(p)!=refs[split]['sha256']:raise ValueError('evidence digest')
        pieces=load_pieces(manifest[split],identity['source']); scores=[]
        with np.load(p) as archive:
            full,short=archive['full'],archive['short']
        for i,piece in enumerate(pieces):
            raw=np.frombuffer(piece,dtype=np.uint8); y=raw[256:2048];qy=probs(piece,256,y);h=histogram(full,i*N)
            if not np.array_equal(mix(qy,h,y,0),qy) or not np.array_equal(mix(qy,0*h,y,1),qy):raise ValueError('off parity')
            if split=='calibration':pred=[mix(qy,h,y,x) for x in GRID]
            else:
                local=np.lib.stride_tricks.sliding_window_view(raw,256)[:N]; uni=((local==y[:,None]).sum(1)+1/256)/257
                pred=[qy,mix(qy,h,y,lam),mix(qy,histogram(short,i*N),y,lam),mix(qy,np.roll(h,1,axis=1),y,lam),probs(piece,2048,y),mix(uni,h,y,1)]
            scores.append([-np.log(x).mean() for x in pred]);
        scores=np.array(scores);means=scores.mean(0)
        if not np.isfinite(means).all():raise ValueError('nonfinite')
        if split=='calibration':
            lam=GRID[means.argmin()];result.update(calibration_grid=list(GRID),calibration_nll=means.tolist(),selected_lambda=lam);print(base,'lambda',lam,flush=True)
        else:
            delta=scores[:,0]-scores[:,1]; ids=np.random.default_rng(3401).integers(0,len(delta),(10000,len(delta)));ci=np.quantile(delta[ids].mean(1),[.025,.975]); result.update(scores=dict(zip(keys,means)),doc_nll=dict(zip(keys,scores.T.tolist())),gain=float(delta.mean()),ci95=ci.tolist(),passed=bool(ci[0]>0 and means[1]<min(means[2:4])),targets=len(delta)*N)
    output.write_text(json.dumps(result,indent=2)+'\n');print(base,result['scores'],flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('base',choices=['gdn2','k15']);a.add_argument('root',type=Path);a.add_argument('output',type=Path);args=a.parse_args();run(args.base,args.root,args.output)
