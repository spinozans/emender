#!/usr/bin/env python3
"""Read-only likelihood diagnostic. No gradients, updates, tools or generation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from ndm.e97 import load_e97_checkpoint
from ndm.e97_agent_server import TorchE97AgentEngine
from ndm.e97_atomic import publish_bytes_no_replace
from scripts.eval_e97_lr_screen import select_cuda_device


def target_logprob(logits,token):
    if logits.ndim!=1 or not torch.isfinite(logits).all():
        raise RuntimeError('invalid or nonfinite next-token logits')
    return float(torch.log_softmax(logits.float(),dim=-1)[token])


def sequence_logprobs(engine,cache,ids):
    values=[]
    for i,token in enumerate(ids):
        values.append(target_logprob(cache.next_logits,token))
        if i+1<len(ids): cache=engine.advance([token],cache)
    return values


@torch.no_grad()
def evaluate(loaded,engine,example,markers,device):
    began=time.monotonic(); prefix=example['prefix_tokens']; target=example['target_tokens']
    cache=engine.advance(prefix)
    serve_logits=cache.next_logits.float()
    top=torch.topk(serve_logits,5).indices.tolist()
    result={'id':example['id'],'split':example['split'],'source':example['source'],
            'prefix_tokens':len(prefix),'target_tokens':len(target),
            'serving_first_top5':[{'id':t,'text':engine.decode([t]),'logprob':target_logprob(serve_logits,t)} for t in top],
            'markers':{}}
    for marker in markers:
        ids=engine.encode(marker)
        logs=sequence_logprobs(engine,cache,ids)
        result['markers'][marker]={'token_ids':ids,'token_logprobs':logs,'logprob_sum':sum(logs)}
    if target:
        logs=sequence_logprobs(engine,cache,target)
        result['serving_target_token_logprobs']=logs
        result['serving_target_nll_mean']=-sum(logs)/len(logs)
        result['serving_first_target_rank']=int((serve_logits>serve_logits[target[0]]).sum())+1
        # The same single-record boundary-aware loss API used by the trainer,
        # without backward or an optimizer. Full packed/gradient parity is NOT
        # established by this shorter-prefix forward diagnostic.
        tokens=torch.tensor([prefix+target],device=device,dtype=torch.long)
        valid=torch.ones_like(tokens,dtype=torch.bool)
        reset=torch.zeros_like(tokens,dtype=torch.bool); reset[:,0]=True
        mask=torch.zeros((1,tokens.shape[1]-1),device=device,dtype=torch.bool)
        mask[:,len(prefix)-1:]=True
        assert int(mask.sum())==len(target)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            loss=loaded.model(tokens,return_loss=True,loss_mask=mask,valid_mask=valid,
                              reset_before=reset,loss_reduction='sum')
        value=float(loss)/len(target)
        assert math.isfinite(value)
        result['boundary_forward_target_nll_mean']=value
        result['forward_minus_serving_nll_mean']=value-result['serving_target_nll_mean']
        del tokens,valid,reset,mask,loss
    # Full-prefix (non-recurrent-call) next-token prediction on the same bytes.
    tokens=torch.tensor([prefix],device=device,dtype=torch.long)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        full=loaded.model(tokens,return_loss=False)
    batch_logits=full[0,-1].detach().float().clone()
    del full,tokens
    assert torch.isfinite(batch_logits).all()
    batch_top=int(batch_logits.argmax())
    result['full_prefix_first_top1']={'id':batch_top,'text':engine.decode([batch_top])}
    result['full_prefix_serving_top1_equal']=batch_top==top[0]
    result['full_prefix_serving_logits_max_abs_difference']=float((batch_logits-serve_logits).abs().max())
    if target: result['full_prefix_first_target_logprob']=target_logprob(batch_logits,target[0])
    result['elapsed_seconds']=time.monotonic()-began
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','args-json','probes','output-root'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True); p.add_argument('--probes-sha256',required=True)
    p.add_argument('--weight-mode',choices=['train','saved'],required=True)
    args=p.parse_args()
    payload=args.probes.read_bytes(); assert hashlib.sha256(payload).hexdigest()==args.probes_sha256
    probes=json.loads(payload); assert probes['schema']=='emender-e97-readonly-learning-probes-v1' and probes['training_eligible'] is False
    with args.checkpoint.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==args.checkpoint_sha256
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    output=args.output_root/f'rank-{rank:02d}.json'
    if output.exists(): raise FileExistsError(output)
    device=select_cuda_device()
    print(f'LEARNING_PROBE_DEVICE rank={rank} device={device}',flush=True)
    loaded=load_e97_checkpoint(args.checkpoint,args_json=args.args_json,device=device,dtype=torch.bfloat16,
                              weight_mode=args.weight_mode,use_triton=True,mmap=True)
    loaded.model.eval()
    engine=TorchE97AgentEngine(loaded,ingest_mode='tokenwise',weight_mode=args.weight_mode,
              device=device,dtype='bfloat16',use_triton=True,checkpoint_sha256=args.checkpoint_sha256,private_analysis=True)
    results=[]
    for example in probes['examples'][rank::world]:
        result=evaluate(loaded,engine,example,probes['marker_candidates'],device)
        results.append(result)
        print(json.dumps({'rank':rank,'id':example['id'],'serving_target_nll':result.get('serving_target_nll_mean'),
                          'top1':result['serving_first_top5'][0]['text']}),flush=True)
    report={'schema':'emender-e97-learning-likelihood-shard-v1','training_eligible':False,
            'checkpoint_sha256':args.checkpoint_sha256,'weight_mode':args.weight_mode,
            'probes_sha256':args.probes_sha256,'rank':rank,'world_size':world,'device':device,
            'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'read-only supervised-prefix likelihood and single-record forward/serving comparison; no backward, training, independent holdout or capability promotion',
            'results':results}
    publish_bytes_no_replace(output,(json.dumps(report,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)
    print(f'LEARNING_PROBE_SHARD_COMPLETE rank={rank}',flush=True)

if __name__=='__main__': main()
