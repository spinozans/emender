#!/usr/bin/env python3
"""Separate read-only forced-opening diagnostic; never an autonomous gate pass."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import tiktoken
from ndm.e97 import load_e97_checkpoint,generate_e97_from_cache
from ndm.e97_agent_server import TorchE97AgentEngine
from ndm.e97_agent_protocol import AgentProtocolError,parse_agent_turn,generated_turn_is_complete
from ndm.e97_atomic import publish_bytes_no_replace
from scripts.eval_e97_lr_screen import evaluate_task,select_cuda_device
from scripts.serve_e97_agent_openai import _archived_controller_source_sha256

PREFILL='Analysis:'


def impossible_prefix(text):
    """Reject only prefixes no appended suffix can make canonically valid."""
    opening='Analysis: "'
    if not (opening.startswith(text) or text.startswith(opening)):
        return True
    first,sep,body=text.partition('\n')
    if not sep: return False
    try:
        parse_agent_turn(first+'\nFinal: diagnostic',private_analysis=True)
    except AgentProtocolError:
        return True
    return not any(start.startswith(body) or body.startswith(start) for start in ('Action: ','Final: '))


class PrefillEngine:
    def __init__(self,inner):
        self.inner=inner
        self.events=[]

    def encode(self,text): return self.inner.encode(text)
    def decode(self,ids): return self.inner.decode(ids)
    def advance(self,ids,cache=None): return self.inner.advance(ids,cache)

    @torch.no_grad()
    def generate(self,cache,*,max_new_tokens,temperature,top_p):
        prefix=self.encode(PREFILL)
        if max_new_tokens<=len(prefix): raise ValueError('budget smaller than prefill')
        if not cache.has_complete_token_history: raise ValueError('diagnostic requires complete token history')
        # Check the actual BPE boundary rather than assuming string concatenation
        # and separately tokenized prefix concatenation are interchangeable.
        if self.encode(self.decode(cache.token_ids)+PREFILL)!=list(cache.token_ids)+prefix:
            raise ValueError('assistant prefill token boundary mismatch')
        unforced=int(cache.next_logits.argmax())
        shadow=self.advance(prefix,cache)
        generated=list(prefix); reason='token_budget'
        for _ in range(max_new_tokens-len(prefix)):
            ids,shadow=generate_e97_from_cache(self.inner.loaded,shadow,max_new_tokens=1,
                temperature=temperature,top_k=0,top_p=top_p,stop_token_ids=(218,))
            if not ids: raise RuntimeError('generation returned no token')
            generated.extend(ids)
            text=self.decode(generated)
            if generated_turn_is_complete(text,private_analysis=True):
                reason='complete_turn'; break
            if impossible_prefix(text):
                reason='provably_invalid_prefix'; break
            if ids[-1]==218:
                reason='record_separator'; break
        self.events.append({'prefill':PREFILL,'prefill_token_ids':prefix,
                            'model_generated_tokens':len(generated)-len(prefix),
                            'unforced_first_token_id':unforced,'unforced_first_token_text':self.decode([unforced]),
                            'stop_reason':reason})
        return generated,shadow


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','args-json','panel','output-root'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True); p.add_argument('--panel-sha256',required=True)
    p.add_argument('--weight-mode',choices=['train','saved'],required=True); args=p.parse_args()
    payload=args.panel.read_bytes(); assert hashlib.sha256(payload).hexdigest()==args.panel_sha256
    panel=json.loads(payload)
    with args.checkpoint.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==args.checkpoint_sha256
    closure=_archived_controller_source_sha256()
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    output=args.output_root/f'rank-{rank:02d}.json'
    if output.exists(): raise FileExistsError(output)
    device=select_cuda_device(); print(f'PREFILL_DEVICE rank={rank} device={device}',flush=True)
    loaded=load_e97_checkpoint(args.checkpoint,args_json=args.args_json,device=device,dtype=torch.bfloat16,
                              weight_mode=args.weight_mode,use_triton=True,mmap=True)
    loaded.model.eval()
    inner=TorchE97AgentEngine(loaded,ingest_mode='tokenwise',weight_mode=args.weight_mode,device=device,
          dtype='bfloat16',use_triton=True,checkpoint_sha256=args.checkpoint_sha256,private_analysis=True)
    engine=PrefillEngine(inner); encoding=tiktoken.get_encoding('p50k_base'); results=[]
    for task in panel['tasks'][rank::world]:
        engine.events=[]
        result=evaluate_task(engine,encoding,panel,task)
        assert len(result['turns'])==len(engine.events)
        for turn,event in zip(result['turns'],engine.events):
            turn['prefill_intervention']=event
            assert turn['completion_tokens']==event['model_generated_tokens']+len(event['prefill_token_ids'])
        result['prefilled_first_turn_protocol_valid']=result.pop('first_turn_protocol_valid')
        results.append(result)
        print(json.dumps({'rank':rank,'task':task['id'],'prefilled_success':result['success'],'error':result['error']}),flush=True)
    report={'schema':'emender-e97-analysis-prefill-shard-v1','training_eligible':False,
            'intervention':'supply Analysis: at every assistant turn; model produces remainder; no parser relaxation',
            'checkpoint_sha256':args.checkpoint_sha256,'weight_mode':args.weight_mode,'panel_sha256':args.panel_sha256,
            'rank':rank,'world_size':world,'device':device,'controller_closure_sha256':closure,
            'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'supplied-opening diagnostic on consumed development instances; not autonomous acquisition, HTTP/Pi qualification, or promotion',
            'results':results}
    publish_bytes_no_replace(output,(json.dumps(report,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)
    print(f'PREFILL_SHARD_COMPLETE rank={rank}',flush=True)

if __name__=='__main__': main()
