"""Frozen weights and frozen soft gates on a hash-bound P100 heldout split."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
from .c11 import Boosted
from .c14_documents import sha
from .c14_paired_eval import strata
from .p100_heldout import ROOT, OUT, DATA, gate, write


def load_models():
    result={};bindings={}
    dense=json.loads((ROOT/'receipts/c14_dense/training.json').read_text())['models']
    for kind,params,index in [('emender',2000000,0),('transformer',2000324,1)]:
        row=dense[index];assert row['arm']==kind
        for label,folder in [('dense',None),('consulted','c14_consultation_emender_v2' if kind=='emender' else 'c14_transformer')]:
            if folder:
                row=json.loads((ROOT/'receipts'/folder/'training.json').read_text());path=Path(row['checkpoint'])
            else:path=Path('/home/erikg/c14_dense_bytes')/(kind+'_dense.pt')
            assert sha(path)==row['checkpoint_sha256']
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model=Boosted(kind,params);model.load_state_dict(saved['model'] if folder else saved)
            result[kind+'_'+label]=model.cuda().eval();bindings[kind+'_'+label]=dict(checkpoint=str(path),sha256=sha(path))
    from .c14_b_control_model import StoreIntegrated
    from .c14_phase2_staged import install_rowkeys
    import gzip
    row=json.loads(gzip.decompress((ROOT/'receipts/c14_b_matched/training.json.gz').read_bytes()))
    path=Path(row['checkpoint']);assert sha(path)==row['checkpoint_sha256']
    binding=json.loads((ROOT/'receipts/scalability_phase2/registration.json').read_text())
    for n,h in binding['sources'].items():assert sha(ROOT.parents[2]/n)==h,n
    install_rowkeys();model=StoreIntegrated('unbudgeted');model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    result['unbounded']=model.cuda().eval();bindings['unbounded']=dict(checkpoint=str(path),sha256=sha(path))
    return result,bindings


def summarize(sums):
    return {k:dict(targets=v[1],finite_nll_sum=v[0],infinite_losses=v[2],nll='Infinity' if v[2] else v[0]/v[1] if v[1] else None) for k,v in sums.items()}

def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python' and os.environ.get('GPU_LEASE_HELD')=='6'
    gate();f=json.loads((OUT/'features.json').read_text());s=json.loads((OUT/'selection.json').read_text())
    assert f['status']=='complete' and sha(DATA/'features.npy')==f['features_sha256'] and sha(DATA/'contexts.npy')==f['contexts_sha256']
    assert sha(DATA/'windows.npz')==s['windows_sha256'] and sha(OUT/'selection.json')==f['selection_sha256']
    assert not (OUT/'eval.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    shapes=json.loads((ROOT/'receipts/c14_byte/model_shapes.json').read_text())
    for path,h in shapes['sources'].items():assert sha(ROOT.parents[2]/path)==h,path
    models,bindings=load_models()
    windows=np.load(DATA/'windows.npz');features=np.load(DATA/'features.npy',mmap_mode='r');contexts=np.load(DATA/'contexts.npy',mmap_mode='r')
    fits={kind:json.loads((ROOT/'receipts'/folder/'frozen_gates.json').read_text()) for kind,folder in [('emender','c14_stacking'),('transformer','c14_transformer_stacking')]}
    gate_bindings={kind:sha(ROOT/'receipts'/folder/'frozen_gates.json') for kind,folder in [('emender','c14_stacking'),('transformer','c14_transformer_stacking')]}
    write('eval_launch.json',dict(checkpoints=bindings,frozen_gate_sha256=gate_bindings,features_sha256=f['features_sha256'],runner_sha256=sha(__file__),in_region_index='Wikipedia100GB',out_of_index_index='Pile-head100GB',limitation='Index/corpus and sample distribution change; coverage difference is descriptive, not causally isolated. Stacked1.5059/1.2636 are soft diagnostics; literal forced walk additionally reported.'))
    sums={}
    with torch.inference_mode():
        for lo in range(0,len(features),16):
            hi=min(lo+16,len(features));xx=torch.as_tensor(windows['x'][lo:hi].astype(np.int64),device='cuda');yy=torch.as_tensor(windows['y'][lo:hi].astype(np.int64),device='cuda');ff=torch.as_tensor(np.array(features[lo:hi]),device='cuda');kk=torch.as_tensor(np.array(contexts[lo:hi]),device='cuda');lengths=windows['lengths'][lo:hi]
            masks=strata(np.array(features[lo:hi]),windows['y'][lo:hi],lengths)
            masks['array_support']=masks.pop('wikipedia_support')
            kval=contexts[lo:hi]
            for name,low,high in [('1_3',1,3),('4_12',4,12),('13_64',13,64),('65_plus',65,1024)]:masks['length_'+name]=masks['total']&(kval>=low)&(kval<=high)
            outputs={};confidence={}
            for label,model in models.items():
                modes=('on','off') if label.endswith('consulted') or label=='unbounded' else ('off',)
                for mode in modes:
                    answers=ff if mode=='on' else torch.zeros_like(ff)
                    kw={}
                    if label=='unbounded':kw=dict(windows=[(w['document'],w['start']) for w in s['windows'][lo:hi]],valid_lengths=lengths.tolist())
                    with torch.autocast('cuda',dtype=torch.bfloat16):lp,_=model(xx,answers,kk,**kw)
                    loss=(-lp.gather(-1,yy[...,None])[...,0]).float().cpu().numpy();key=label+'_'+mode
                    outputs[key]=loss;confidence[key]=lp.float().exp().amax(-1).cpu().numpy()
            q=features[lo:hi,:,:256];qtarget=np.take_along_axis(q,windows['y'][lo:hi,:,None],axis=2)[:,:,0].astype(np.float64);support=features[lo:hi,:,256]>0;fork=features[lo:hi,:,258]>0
            with np.errstate(divide='ignore'):outputs['array_alone']=-np.log(qtarget)
            for kind in fits:
                for source,fitkey in [('dense','dense'),('consulted','v2_on')]:
                    key=kind+'_'+source+'_'+('on' if source=='consulted' else 'off');fit=fits[kind][fitkey]
                    x=np.stack([np.ones(qtarget.shape),confidence[key],features[lo:hi,:,259],features[lo:hi,:,256]],axis=-1).astype(np.float64)
                    x[:,:,1:]=(x[:,:,1:]-fit['center'])/fit['scale'];g=1/(1+np.exp(-np.clip(x@np.asarray(fit['weights']),-35,35)))
                    p=np.exp(-outputs[key].astype(np.float64));soft=np.where(support,(1-g)*p+g*qtarget,p)
                    walk=np.where(support&~fork,qtarget,soft)
                    with np.errstate(divide='ignore'):
                        outputs[kind+'_'+source+'_stacked_soft']=-np.log(soft);outputs[kind+'_'+source+'_stacked_literal']=-np.log(walk)
            for arm,loss in outputs.items():
                for name,mask in masks.items():
                    values=loss[mask].astype(np.float64);assert not np.isnan(values).any() and not np.isneginf(values).any()
                    total=sums.setdefault(arm+'/'+name,[0.,0,0]);finite=np.isfinite(values);total[0]+=float(values[finite].sum());total[1]+=len(values);total[2]+=int((~finite).sum())
            write('eval_progress.json',dict(status='running',next_window=hi,table=summarize(sums)));print('eval',hi,flush=True)
    table=summarize(sums);assert all(v['targets']==s['targets'] for k,v in table.items() if k.endswith('/total'))
    gains={}
    for label in ('emender_consulted','transformer_consulted','unbounded'):
        off=table[label+'_off/total']['nll'];on=table[label+'_on/total']['nll'];gains[label]=dict(off_nll=off,on_nll=on,transfer_gain=off-on)
    for kind in fits:
        gains[kind+'_stacked']=dict(transfer_gain=table[kind+'_consulted_off/total']['nll']-table[kind+'_consulted_stacked_soft/total']['nll'])
    write('eval.json',dict(status='complete',windows=len(features),targets=s['targets'],documents=len(s['documents']),table=table,gains=gains,eval_launch_sha256=sha(OUT/'eval_launch.json'),single_seed=True,source_shift_confound=True,training_steps=0))

if __name__=='__main__':main()
