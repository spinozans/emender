"""Independent substring feature audit and raw-loss/denominator receipt replay."""
import json
from pathlib import Path
import numpy as np
from .c12 import ROOT, OUT, digest

def main():
    corpora=json.loads((OUT/'corpora.json').read_text())
    for v in corpora.values():
        if isinstance(v,dict):assert digest(ROOT/v['path'])==v['sha256']
    x=(OUT/'corpus_x.txt').read_bytes();y=(ROOT/'input.txt').read_bytes()
    z=np.load(ROOT/'receipts/c6/index.npz');d=np.load(OUT/'disjoint.npz');dev=json.loads((OUT/'device.json').read_text())
    assert dev['index_sha256']==digest(OUT/'disjoint.npz')
    assert dev['corpora_sha256']==digest(OUT/'corpora.json')
    assert dev['source_sha256']==digest(ROOT/'c12.py')
    rng=np.random.default_rng(120043);sample=np.unique(np.r_[0,len(y)-1,rng.integers(0,len(y),2048)])
    for i in sample:
        l=int(z['context_length'][i]);context=y[i-l:i];c=np.zeros(256,dtype=np.int64)
        if not context:c=np.bincount(np.frombuffer(x,dtype=np.uint8),minlength=256)
        else:
            start=0
            while True:
                hit=x.find(context,start)
                if hit<0:break
                target=hit+len(context)
                if target<len(x):c[x[target]]+=1
                start=hit+1
        row=d['features'][d['ids'][i]];total=c.sum();p=c/total if total else np.zeros(256)
        expected=np.r_[p,np.log1p(total),-(p[p>0]*np.log(p[p>0])).sum(),np.count_nonzero(c)>1,np.log1p(l)]
        assert np.allclose(row,expected,rtol=2e-6,atol=1e-6),i
        assert d['support'][i]==np.count_nonzero(c)
        assert d['target_supported'][i]==(c[y[i]]>0)
    meta=json.loads((OUT/'training.json').read_text());results=json.loads((OUT/'results.json').read_text())
    assert meta['protocol_sha256']==digest(ROOT/'PREREGISTRATION_C12.md')==results['protocol_sha256']
    assert meta['source_sha256']==digest(ROOT/'c12.py') and meta['physical_gpu'] in ('2','4')
    assert len(meta['models'])==4 and not meta['failures']
    for p,h in meta['inherited_code'].items():assert digest(ROOT/p)==h
    for row in meta['models']:
        assert row['training_seconds']>=180 and row['tokens']==row['steps']*2048
        if row['keep_fraction'] is not None:assert .49<row['keep_fraction']<.51
        assert row['prediction_sha256']==digest(OUT/(row['name']+'_prediction.npz'))
    allpos=np.arange(int(.9*len(y))+1,len(y));positions=allpos[len(allpos)//2:]
    masks=dict(total=np.ones(len(positions),bool),fork=z['fork'][positions],ood_x=d['support'][positions]==0,
        ood_self=z['support'][positions]==0,x_fork=d['support'][positions]>1,
        x_nonempty_zero_target=(d['support'][positions]>0)&~d['target_supported'][positions],
        self_nonempty_zero_target=(z['support'][positions]>0)&(z['q'][positions]==0))
    checks=0
    for name,input in results['inputs'].items():
        path=ROOT/input['path'];assert digest(path)==input['sha256'];ev=np.load(path)
        model_name=name
        if name.endswith('_c11_replay'):
            model_name=name.removesuffix('_replay');losses={'without':ev['without']}
            reference=np.load(ROOT/'receipts/c11'/(name.split('_')[0]+'_ops_prediction.npz'))['p']
            assert np.allclose(np.exp(-ev['with_replayed'].astype(np.float32)),reference,rtol=1e-5,atol=1e-7)
        elif name.endswith('_dense'):losses={'without':-np.log(ev['p'])}
        elif name.endswith('_c11'):losses={'with':-np.log(ev['p'])}
        else:losses=dict(ev)
        for mode,loss in losses.items():
            assert len(loss)==len(allpos) and np.isfinite(loss).all()
            for col,mask in masks.items():
                want=results['models'][model_name][mode][col];assert want['n']==int(mask.sum())
                if mask.any():assert abs(want['nll']-float(loss[len(allpos)//2:][mask].sum()/mask.sum()))<1e-12
                else:assert want['nll'] is None
                checks+=1
    replay=json.loads((OUT/'control_replay.json').read_text())
    assert replay['protocol_sha256']==digest(ROOT/'PREREGISTRATION_C12_CONTROL_REPLAY.md')
    assert replay['source_sha256']==digest(ROOT/'c12_control_replay.py')
    assert json.loads((OUT/'replay_terminal.json').read_text())['exit_code']==0
    assert json.loads((OUT/'terminal.json').read_text())['exit_code']==0
    print(json.dumps(dict(status='green',independent_substring_feature_checks=len(sample),loss_denominator_checks=checks,
        corpora_hashes='matched',budgets='four >=180s',gpu=meta['physical_gpu'],scope='not a significance test or universal no-skill proof')))
if __name__=='__main__':main()
