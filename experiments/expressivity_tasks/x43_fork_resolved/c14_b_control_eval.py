"""Frozen-window PB1--PB4 evaluation of the selected, registered B1 checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
from .c14_b_model import StoreIntegrated
from . import c14_b_control_fast as fast
from .c14_b_oracle import oracle_shard
from .c14_b_eval_decode import load_context_decoder,eval_shard
from .c14_b_train import ROOT,OUT,CODE,accumulate
from .c14_documents import sha
from .c14_fast_features import load_decoder
from .c14_paired_eval import CACHE,DATA,JOURNAL,WINDOWS,EXPECTED,strata,add_sums,mean


def write(name,value):
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def synthetic_decay(model):
    from .c14_b_native import native_forward,native_reports
    from .c14_b_suffix import queried_suffix
    result=[]
    for layer,block in enumerate(model.head.base.blocks):
        for h,cfg in enumerate(block.cell.configs):
            ticks=1024
            raw=[b'a'*ticks];windows=[(0,0)]
            owner=queried_suffix(windows[0],b'a',1)
            metadata=([[owner]*ticks],raw,windows,[ticks],[[True]+[False]*(ticks-1)],[[97]*ticks])
            inputs=[torch.zeros(1,ticks,d,device='cuda') for d in (64,64,128,128)]
            inputs[3].fill_(1.)
            inputs += [torch.zeros(1,ticks,device='cuda'),torch.full((1,ticks),100.,device='cuda'),torch.full((1,ticks),-100.,device='cuda')]
            inputs += [torch.tensor(0.,device='cuda') for _ in range(3)]
            _,(state,trace)=native_forward(inputs,cfg,block.cell.gate_policies[h],metadata)
            stats=native_reports(trace,cfg,[ticks])[0]
            after=trace[0,:,14].cpu().tolist()
            empty=next(i+1 for i,n in enumerate(after) if n==0)
            assert stats['APPEND']==1 and stats['natural_retirements']==1 and stats['final_array']==0
            assert stats['stale_retirements']==0 and stats['final_stream']==0
            result.append(dict(layer=layer,head=h,initial_array_slots=1,ticks_to_empty=empty,
                decay_rate=cfg.decay_rate,free_threshold=cfg.free_threshold,pop_threshold=cfg.pop_threshold,
                residual_array_slots=0,stream_slots_aged=0,native_operation_counts=stats['DECAY']))
    return result


def verdict(state,training,decay):
    table={k:dict(total_nll=s,targets=n,nll=s/n if n else None) for k,(s,n) in state['sums'].items()}
    def metric(mode,name='total'):return table[f'{mode}/final_test/{name}']
    total=metric('B1_on');delta=total['nll']-1.509260010
    assert total['targets']==214097022
    frozen=json.loads((ROOT/'receipts/c14_paired_eval/table.json').read_text())
    pb2={}
    for name,reference in [('supported_target',.811639725),('zero_target',3.517122903),('fork',2.010901269)]:
        row=metric('B1_on',name);control=frozen[f'v2_on/final_test/{name}']
        assert row['targets']==control['targets']
        pb2[name]=dict(**row,v2_reference=reference,delta=row['nll']-reference)
    off=metric('B1_off');dense=frozen['dense/final_test/total']
    assert off['targets']==dense['targets']
    unplug_delta=off['nll']-dense['nll']
    return dict(status='complete',PB1=dict(**total,v2_reference=1.509260010,delta=delta,
            outcome='store door better' if delta<0 else 'door indifferent at measured precision' if delta==0 else 'input channel better',
            dense_reference=2.276133549,delta_vs_dense=total['nll']-2.276133549),PB2=pb2,
        PB3=dict(training=training['telemetry'],evaluation_sampled=state['telemetry'],
            findability='nonzero effective float32 read weights / live array slot read opportunities; not top-1 or causal-benefit evidence',
            absent_evidence=[key+'/'+op for key,row in training['telemetry'].items() for op in ('CORRECT','MERGE') if row['counts'][op]==0]),
        PB4=dict(**off,dense_control_nll=dense['nll'],nll_delta=unplug_delta,absolute_nll_delta=abs(unplug_delta),
            graceful_margin=.02,within_margin=abs(unplug_delta)<=.02,reset_1_16=metric('B1_off','window_reset_1_16'),
            synthetic_populated_store_native_decay=decay,new_array_insertions=0,
            reset_state='same independent empty window reset; zero aging ticks needed at held-out reset'),
        training={k:training[k] for k in ('steps','actual_bytes_targets','active_seconds','status','checkpoint_sha256')},
        conditional_ablations_required=False,conditional_ablations='registered conditional control; no recursive ablations',
        calibration_separate=True,no_live_queries=True)


def main():
    assert sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    training=json.loads((OUT/'training.json').read_text())
    assert training['status']=='completed_active_budget'
    for name,digest in training['config']['code_sha256'].items():assert sha(ROOT/name)==digest,name
    checkpoint=Path(training['checkpoint']);assert sha(checkpoint)==training['checkpoint_sha256']
    assert sha(CACHE/'journal.jsonl')==JOURNAL
    checked=json.loads((ROOT/'receipts/c14_paired_eval/cache_validation.json').read_text())
    assert checked['status']=='passed' and checked['journal_sha256']==JOURNAL and checked['all_eval_entropy_float32_identity']
    construction=json.loads((ROOT/'receipts/c14_dense/eval_construction.json').read_text())
    assert sha(Path(construction['windows_path']))==WINDOWS
    p=json.loads((ROOT/'receipts/c14_dense/pile_span.json').read_text())
    assert sha(DATA/'pile_offsets.jsonl')==p['offsets_sha256']
    source=Path(p['source']);stat=source.stat()
    assert (stat.st_size,stat.st_ino,stat.st_mtime_ns)==(p['source_size'],p['source_inode'],p['source_mtime_ns'])
    raw=np.memmap(source,dtype=np.uint8,mode='r')
    docs={d['source_index']:d for line in (DATA/'pile_offsets.jsonl').open() if (d:=json.loads(line))['split']=='valid'}
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=StoreIntegrated().cuda().eval();model.load_state_dict(saved['model'])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    decoder=load_decoder();context_decoder=load_context_decoder()
    config=dict(checkpoint_sha256=training['checkpoint_sha256'],eval_code_sha256=sha(__file__),
        code_sha256=training['config']['code_sha256'],journal_sha256=JOURNAL,windows_sha256=WINDOWS,
        metadata_reuse_sha256=sha(ROOT/'c14_b_control_fast.py'),
        metadata_decoder_sha256=sha(ROOT/'c14_b_contexts.c'),metadata_wrapper_sha256=sha(ROOT/'c14_b_eval_decode.py'),
        precision='BF16 autocast; per-target CE float32; sums float64',deterministic=True,
        modes=['B1_on','B1_off'],reset='independent window; document tails loss masked')
    if args.resume:
        state=json.loads((OUT/'eval_resume.json').read_text());assert state['config']==config
    else:
        assert not (OUT/'eval_resume.json').exists(),'use --resume'
        state=dict(config=config,next_shard=0,windows=0,targets=0,sums={},telemetry={},elapsed_seconds=0.,status='running')
    decay=synthetic_decay(model);write('native_populated_decay.json',decay)
    shards=[r for s in (CACHE/'journal.jsonl').read_text().splitlines() if (r:=json.loads(s))['event']=='shard_complete']
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    fast.install()
    start=time.monotonic();prior=state['elapsed_seconds'];verified=set()
    def save(status):
        state['elapsed_seconds']=prior+time.monotonic()-start;state['status']=status;write('eval_resume.json',state)
    with Path(construction['windows_path']).open() as frozen,torch.inference_mode():
        for _ in range(state['windows']):next(frozen)
        for index in range(state['next_shard'],len(shards)):
            r=shards[index];data=(CACHE/r['file']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==r['sha256']
            ids,lengths,features,contexts=eval_shard(data,r['windows'],decoder,context_decoder)
            if index in (0,len(shards)//2,len(shards)-1):
                reference=oracle_shard(data,r['windows'],decoder)
                for actual,expected in zip((ids,lengths,features,contexts),reference[:4]):
                    assert np.array_equal(actual,expected),'eval oracle/raw metadata mismatch'
            assert r['first_window_index']==state['windows']
            xx=np.zeros((len(ids),128),np.int64);yy=np.zeros_like(xx);halves=[]
            for j,((doc,at),length) in enumerate(zip(ids.tolist(),lengths.tolist())):
                expected=json.loads(next(frozen))
                assert (doc,at,length)==(expected['document'],expected['start'],expected['targets'])
                d=docs[doc]
                if doc not in verified:
                    assert hashlib.sha256(raw[d['byte_offset']:d['byte_offset']+d['byte_length']]).hexdigest()==d['sha256']
                    verified.add(doc)
                assert at+length<d['byte_length']
                begin=d['byte_offset']+at;bb=raw[begin:begin+length+1]
                xx[j,:length]=bb[:-1];yy[j,:length]=bb[1:];halves.append(expected['half'])
            masks=strata(features,yy,lengths);halves=np.array(halves)
            x=torch.as_tensor(xx,device='cuda');y=torch.as_tensor(yy,device='cuda')
            answers=torch.as_tensor(features,device='cuda');ll=torch.as_tensor(contexts,device='cuda')
            for mode in config['modes']:
                fast.full_telemetry=(index%64==0 and mode=='B1_on')
                ff=answers if mode=='B1_on' else torch.zeros_like(answers)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    lp,_=model(x,ff,ll,windows=[tuple(row) for row in ids.tolist()],valid_lengths=lengths.tolist())
                if index==0:
                    fast.restore()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        reference,_=model(x,ff,ll,windows=[tuple(row) for row in ids.tolist()],valid_lengths=lengths.tolist())
                    assert torch.equal(lp,reference),'eval metadata/telemetry optimization changed logits'
                    fast.install()
                losses=(-lp.gather(-1,y[...,None])[...,0]).float().cpu().numpy()
                assert np.isfinite(losses).all(),(mode,index)
                if mode=='B1_on' and index%64==0:accumulate(state['telemetry'],model.telemetry)
                if mode=='B1_off':assert sum(row['array_requests'] for layer in model.telemetry for row in layer)==0
                for half in ('all','calibration','final_test'):
                    selection=np.ones(len(ids),bool) if half=='all' else halves==half
                    add_sums(state['sums'],mode,half,losses,masks,selection)
            state['next_shard']=index+1;state['windows']+=len(ids);state['targets']+=int(lengths.sum())
            if index%64==0 or stopped:
                save('running');print(json.dumps(dict(shard=index+1,windows=state['windows'],seconds=state['elapsed_seconds'],
                    final_nll=mean(state['sums'].get('B1_on/final_test/total',[0.,0])))),flush=True)
            if stopped:save('paused');return
        assert frozen.read()==''
    assert (state['windows'],state['targets'])==EXPECTED[:2]
    save('complete')
    write('eval_table.json',{k:dict(total_nll=s,targets=n,nll=s/n if n else None) for k,(s,n) in state['sums'].items()})
    write('verdicts.json',verdict(state,training,decay))


if __name__=='__main__':main()
