#!/usr/bin/env python3
"""Bounded layer traces and execution-layout ablations; never changes model weights."""
import argparse
import json
import os
from pathlib import Path
from scripts.eval_e97_native_execution import publish,sha
from scripts.qualify_e97_native_rl_logprobs import turn_layout

ROOT=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
SOURCE=ROOT/'native-rl-logprob-qualification-v1/recipe-private.json'
SOURCE_SHA='ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f'
KEYS=[('onpolicy-task-000-sample-0',0),('onpolicy-task-004-sample-0',1),
      ('onpolicy-task-010-sample-0',1),('onpolicy-task-014-sample-0',4)]
PROFILES=[
    dict(name='actor-cache',segmented=True,train=False,amp=False,loss=False,padded=False,checkpoint=False,mlp=0),
    dict(name='actor-amp',segmented=True,train=False,amp=True,loss=False,padded=False,checkpoint=False,mlp=0),
    dict(name='full-eval',segmented=False,train=False,amp=False,loss=False,padded=False,checkpoint=False,mlp=0),
    dict(name='full-train',segmented=False,train=True,amp=False,loss=False,padded=False,checkpoint=False,mlp=0),
    dict(name='full-train-amp',segmented=False,train=True,amp=True,loss=False,padded=False,checkpoint=False,mlp=0),
    dict(name='masked-unpadded',segmented=False,train=True,amp=True,loss=True,padded=False,checkpoint=False,mlp=0),
    dict(name='masked-padded',segmented=False,train=True,amp=True,loss=True,padded=True,checkpoint=False,mlp=0),
    dict(name='checkpointed',segmented=False,train=True,amp=True,loss=True,padded=True,checkpoint=True,mlp=0),
    dict(name='training-default',segmented=False,train=True,amp=True,loss=True,padded=True,checkpoint=True,mlp=4096),
]
PAIRS=[('actor-cache','actor-amp'),('actor-cache','full-eval'),('full-eval','full-train'),
       ('full-train','full-train-amp'),('full-train-amp','masked-unpadded'),
       ('masked-unpadded','masked-padded'),('masked-padded','checkpointed'),('checkpointed','training-default')]


def freeze(args):
    if sha(SOURCE)!=SOURCE_SHA:raise ValueError('source recipe identity')
    source=json.loads(SOURCE.read_text());table={(r['id'],r['turn']):r for r in source['selected']}
    selected=[table[k] for k in KEYS]
    recipe=dict(schema='emender-e97-activation-alignment-v1',model=source['model'],args_json=source['args_json'],
        args_sha256=source['args_sha256'],selected=selected,profiles=PROFILES,pairs=PAIRS,
        source_recipe_sha256=SOURCE_SHA,selection='Four previously inspected control/outlier turns; diagnostic, not independent validation',
        reference_measurements=str(ROOT/'native-rl-logprob-reproduction-v2/measurements-private.json'),
        reference_measurements_sha256='a2e3399ba8974310db27df9094596237e16a2459050e3f8f8169bd197d6dc9f9',
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,workers=1,
        parameter_sha256='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd',
        maximum_prefix_tokens=16384,maximum_generated_tokens=512,cpu_trace_limit_bytes=3*1024**3,
        repetitions_per_profile=2,endpoint_binding_limit=1e-4,
        scope='No gradients, updates or activation replacement; layout localization only; all existing failed gates unchanged')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False);publish(args.output/'recipe-private.json',recipe)
    print('ACTIVATION_DIAGNOSTIC_FROZEN',sha(args.output/'recipe-private.json'),flush=True)


def load(args):
    if sha(args.recipe)!=args.recipe_sha:raise ValueError('recipe identity')
    r=json.loads(args.recipe.read_text())
    if r['profiles']!=PROFILES or r['pairs']!=[list(x) for x in PAIRS] or r['workers']!=1:
        raise ValueError('execution recipe mismatch')
    if [(x['id'],x['turn']) for x in r['selected']]!=KEYS:raise ValueError('selection mismatch')
    if r['endpoint_binding_limit']!=1e-4 or r['repetitions_per_profile']!=2:raise ValueError('binding/repetition recipe mismatch')
    return r


class Taps:
    """Select prediction rows only. All hooks return None; no tensor replacement."""
    def __init__(self,model):
        self.model=model;self.bank={};self.handles=[];self.positions=[];self.targets=[];self.head_offset=0;self.logprobs=[];self.bytes=0
        self.handles.append(model.embedding.register_forward_hook(self.output('embedding')))
        for i,layer in enumerate(model.layers):
            self.handles.append(layer.register_forward_pre_hook(self.input(f'{i:02d}.input')))
            self.handles.append(layer.mixer.register_forward_hook(self.output(f'{i:02d}.mixer')))
            self.handles.append(layer.mlp.register_forward_pre_hook(self.input(f'{i:02d}.mlp-input')))
            self.handles.append(layer.mlp.register_forward_hook(self.output(f'{i:02d}.mlp-output')))
            self.handles.append(layer.register_forward_hook(self.output(f'{i:02d}.output')))
        self.handles.append(model.lm_head.register_forward_hook(self.head))

    def begin(self,positions,targets):
        if len(positions)!=len(targets) or len(set(positions))!=len(positions) or list(positions)!=sorted(positions) or any(p<0 for p in positions):
            raise ValueError('prediction row mapping')
        self.positions=list(positions);self.targets=list(targets);self.head_offset=0

    def store(self,name,tensor,positions):
        import torch
        if not positions:return
        if tensor.ndim!=3 or tensor.shape[0]!=1 or max(positions)>=tensor.shape[1]:
            raise ValueError('activation row shape/coverage')
        if tensor.shape[-1]>3840 or len(positions)>512:raise ValueError('activation feature/row bound')
        self.bytes+=len(positions)*tensor.shape[-1]*tensor.element_size()
        if self.bytes>3*1024**3//6:raise ValueError('CPU trace storage bound')
        rows=tensor[0,positions].detach().cpu().clone()
        if not bool(torch.isfinite(rows).all()):raise ValueError('nonfinite activation')
        self.bank.setdefault(name,[]).append(rows)

    def input(self,name):
        def hook(module,inputs):self.store(name,inputs[0],self.positions)
        return hook

    def output(self,name):
        def hook(module,inputs,output):self.store(name,output[0] if isinstance(output,tuple) else output,self.positions)
        return hook

    def head(self,module,inputs,output):
        import torch
        start=self.head_offset;stop=start+output.shape[1]
        chosen=[(p-start,t) for p,t in zip(self.positions,self.targets) if start<=p<stop]
        positions=[p for p,_ in chosen];self.store('head-input',inputs[0],positions)
        # Bound temporary FP32 probability rows, even for a full-sequence head.
        for offset in range(0,len(chosen),128):
            part=chosen[offset:offset+128]
            targets=torch.tensor([t for _,t in part],device=output.device,dtype=torch.long)
            logits=output[0,[p for p,_ in part]].float()
            values=torch.log_softmax(logits,-1).gather(1,targets[:,None]).squeeze(1)
            if not bool(torch.isfinite(values).all()):raise ValueError('nonfinite probability')
            self.logprobs.extend(values.cpu().tolist())
        self.head_offset=stop

    def finish(self,count):
        import torch
        expected=['embedding']+[f'{i:02d}.{stage}' for i in range(len(self.model.layers))
                    for stage in ('input','mixer','mlp-input','mlp-output','output')]+['head-input']
        if set(self.bank)!=set(expected) or len(self.logprobs)!=count:raise ValueError('trace stage/target coverage')
        result={k:torch.cat(self.bank[k]) for k in expected}
        if any(v.ndim!=2 or len(v)!=count for v in result.values()):raise ValueError('trace row coverage')
        return result

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]


def compare_banks(reference,current):
    import hashlib
    import torch
    if list(reference)!=list(current):raise ValueError('trace stage identity')
    rows=[]
    for name,a in reference.items():
        b=current[name]
        if a.shape!=b.shape:raise ValueError('trace shape identity')
        delta=b.float()-a.float();absolute=delta.abs().amax(dim=1)
        relative=delta.norm(dim=1)/a.float().norm(dim=1).clamp_min(1e-30)
        if not bool(torch.isfinite(relative).all() & torch.isfinite(absolute).all()):raise ValueError('nonfinite trace comparison')
        rows.append(dict(stage=name,reference_dtype=str(a.dtype),current_dtype=str(b.dtype),dtype_changed=a.dtype!=b.dtype,
                         absolute_max=float(absolute.max()),relative_l2_max=float(relative.max()),
                         differing_elements=int(torch.count_nonzero(delta)),absolute_by_position=absolute.tolist(),relative_by_position=relative.tolist(),
                         reference_sha256=hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                         current_sha256=hashlib.sha256(b.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
    return rows


def evaluate(loaded,item,profile):
    import torch
    from ndm.e97 import advance_e97_cache_segment,advance_e97_cache
    model=loaded.model;model.train(profile['train']);model.gradient_checkpointing=profile['checkpoint']
    model.gradient_checkpoint_group_size=3;model.loss_chunk_size=128;model.loss_logits_fp32=True;model.checkpoint_loss_chunks=True
    for layer in model.layers:layer.mlp.checkpoint_chunk_size=profile['mlp']
    prefix=item['prefix'];generated=item['generated'];device=next(model.parameters()).device
    taps=Taps(model)
    try:
        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=profile['amp']):
            if profile['segmented']:
                taps.begin([len(prefix)-1],[generated[0]])
                cache=advance_e97_cache_segment(loaded,prefix)
                for i,token in enumerate(generated):
                    taps.begin([0] if i+1<len(generated) else [],[generated[i+1]] if i+1<len(generated) else [])
                    cache=advance_e97_cache(loaded,[token],cache)
            else:
                positions=list(range(len(prefix)-1,len(prefix)+len(generated)-1));taps.begin(positions,generated)
                if profile['loss']:
                    tokens,valid,reset,mask=turn_layout(prefix,generated,device,128 if profile['padded'] else 1)
                    if tokens[0,positions].tolist()!=(prefix+generated[:-1])[len(prefix)-1:]:raise ValueError('prediction input identity')
                    model(tokens,return_loss=True,valid_mask=valid,reset_before=reset,loss_mask=mask,loss_reduction='sum')
                else:
                    tokens=torch.tensor([prefix+generated[:-1]],device=device,dtype=torch.long)
                    model(tokens,return_loss=False)
        bank=taps.finish(len(generated));return bank,taps.logprobs
    finally:taps.close()


def run(args):
    import torch
    from ndm.e97 import load_e97_checkpoint
    from scripts.audit_e97_live_actor_capture import fingerprint,runtime
    recipe=load(args);target=recipe['model'];local=int(os.environ.get('LOCAL_RANK','0'))
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single fixed worker')
    for path,digest in ((target['checkpoint'],target['sha256']),(recipe['args_json'],recipe['args_sha256']),
                        (recipe['reference_measurements'],recipe['reference_measurements_sha256'])):
        if sha(path)!=digest:raise ValueError('input identity')
    torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':raise ValueError('precision defaults')
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=recipe['args_json'],device=torch.device('cuda',local),
                               dtype=torch.bfloat16,weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model
    if len(model.layers)!=18 or any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in model.modules()):raise ValueError('expected deterministic 18-layer model')
    if any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('persistent dtype')
    before=fingerprint(model);metadata=runtime(loaded,local)
    if before['parameters']!=recipe['parameter_sha256']:raise ValueError('effective parameter identity')
    references={(r['id'],r['turn']):r for r in json.loads(Path(recipe['reference_measurements']).read_text())['records']}
    reports=[]
    for item in recipe['selected']:
        if not 0<len(item['prefix'])<=recipe['maximum_prefix_tokens'] or not 0<len(item['generated'])<=recipe['maximum_generated_tokens']:
            raise ValueError('trace budget')
        if item['recorded_logprobs']!=references[(item['id'],item['turn'])]['recorded']:raise ValueError('endpoint reference identity')
        actor=None;previous=None;actor_lp=None;previous_lp=None;case=[]
        for profile in PROFILES:
            torch.manual_seed(974223)
            bank,lp=evaluate(loaded,item,profile)
            if actor is None:actor=bank;actor_lp=lp
            if sum(v.numel()*v.element_size() for v in bank.values())*6>recipe['cpu_trace_limit_bytes']:raise ValueError('CPU trace budget')
            torch.manual_seed(974223)
            repeated,repeated_lp=evaluate(loaded,item,profile)
            repeat_comparison=compare_banks(bank,repeated)
            repeat_equal=all(r['reference_sha256']==r['current_sha256'] and not r['dtype_changed'] for r in repeat_comparison)
            del repeated
            against_actor=compare_banks(actor,bank)
            pair=next((p for p in PAIRS if p[1]==profile['name']),None)
            pair_reference=actor if pair is not None and pair[0]=='actor-cache' else previous
            pair_lp=actor_lp if pair is not None and pair[0]=='actor-cache' else previous_lp
            compared=compare_banks(pair_reference,bank) if pair is not None else []
            original=references[(item['id'],item['turn'])]
            reference_lp=original['teacher'] if profile['name']=='training-default' else item['recorded_logprobs']
            row=dict(profile=profile,logprobs=lp,against_actor=against_actor,isolated_pair=pair,pair_comparison=compared,
                     repeat_comparison=repeat_comparison,repeat_activations_equal=repeat_equal,
                     repeat_logprob_max_delta=max(abs(a-b) for a,b in zip(lp,repeated_lp)),
                     actor_max_delta=max(abs(a-b) for a,b in zip(lp,actor_lp)),
                     pair_max_delta=max(abs(a-b) for a,b in zip(lp,pair_lp)) if pair is not None else None,
                     first_differing_stage=next((x['stage'] for x in compared if x['differing_elements'] or x['dtype_changed']),None),
                     endpoint_reference_max_delta=max(abs(a-b) for a,b in zip(lp,reference_lp)) if profile['name'] in ('actor-cache','training-default') else None)
            case.append(row);previous=bank;previous_lp=lp
            print('ACTIVATION_PROFILE_COMPLETE',item['id'],item['turn'],profile['name'],row['first_differing_stage'],flush=True)
        reports.append(dict(id=item['id'],turn=item['turn'],profiles=case))
    after=fingerprint(model)
    if before!=after or any(p.grad is not None for p in model.parameters()):raise ValueError('model mutation or unexpected gradients')
    publish(args.output/'measurements-private.json',dict(recipe_sha256=args.recipe_sha,reports=reports))
    bound=all(p['endpoint_reference_max_delta']<=recipe['endpoint_binding_limit'] for r in reports for p in r['profiles'] if p['endpoint_reference_max_delta'] is not None)
    repeatable=all(p['repeat_activations_equal'] and p['repeat_logprob_max_delta']<=recipe['endpoint_binding_limit'] for r in reports for p in r['profiles'])
    result=dict(status='diagnostic-measurements-complete',localization_ready=bound and repeatable,repeatability_passed=repeatable,
                endpoint_binding_passed=bound,endpoint_binding_limit=recipe['endpoint_binding_limit'],recipe_sha256=args.recipe_sha,fingerprint=before,
                runtime=metadata,optimizer_updates=0,training_eligible=False,rl_optimizer_ready=False,automatic_expansion=False,
                original_probability_gate='failed, unchanged',peak_hbm_allocated=torch.cuda.max_memory_allocated(local),
                cases=[dict(id=r['id'],turn=r['turn'],profiles=[{k:v for k,v in p.items() if k not in ('logprobs','against_actor','pair_comparison','repeat_comparison')} for p in r['profiles']]) for r in reports])
    publish(args.output/'summary.json',result);print('ACTIVATION_DIAGNOSTIC_COMPLETE',sha(args.output/'summary.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('run');q.add_argument('--output',type=Path,required=True);q.add_argument('--recipe',type=Path,required=True);q.add_argument('--recipe-sha',required=True)
    a=p.parse_args();globals()[a.command](a)
