#!/usr/bin/env python3
"""No-update actor replay versus masked CE128 training-layout likelihood assay."""
import argparse
import json
import os
from pathlib import Path

from scripts.eval_e97_native_execution import publish, sha

SOURCE='/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-onpolicy-canary-v2'
SUMMARY_SHA='8278969f8ff65afd6f49251abb8a05c758ef6586c748824d3cdbc8345e187e6d'


def freeze(args):
    root=Path(SOURCE)
    if sha(root/'results/summary.json')!=SUMMARY_SHA:raise ValueError('rollout summary identity')
    summary=json.loads((root/'results/summary.json').read_text())
    panel=json.loads((root/'panel.json').read_text())
    if sha(root/'panel.json')!=summary['panel_sha256']:raise ValueError('rollout panel identity')
    selected=[]
    for row in summary['results']:
        if not row['id'].endswith('-sample-0'):continue
        path=root/'results'/row['id']/'episode-private.json'
        if sha(path)!=row['episode_sha256']:raise ValueError('episode identity')
        episode=json.loads(path.read_text())
        for i,turn in enumerate(episode['generations']):
            trace=turn['sampling'];ids=turn['token_ids'];prefix=trace['prompt_token_ids']
            if (trace['temperature'],trace['top_k'],trace['top_p'])!=(1.,0,0.):raise ValueError('behavior distribution')
            if not ids or len(ids)>512 or len(prefix)>16384 or len(ids)!=len(trace['selected_logprobs']):
                raise ValueError('all-turn no-filter audit bounds/coverage')
            selected.append(dict(id=row['id'],turn=i,episode_sha256=row['episode_sha256'],
                                 prefix=prefix,generated=ids,recorded_logprobs=trace['selected_logprobs']))
    if len({s['id'] for s in selected})!=16 or len(selected)>128:raise ValueError('all sample-zero episodes required')
    recipe=dict(schema='emender-native-rl-logprob-assay-v1',model=panel['models'][0],
                args_json=panel['args_json'],args_sha256=panel['args_sha256'],
                source_summary_sha256=SUMMARY_SHA,selected=selected,
                selection='Every generated turn from sample 0 of all 16 training tasks; no filtering',
                cached_replay_abs_max=1e-4,teacher_abs_max=.05,teacher_abs_p99=.02,ce_mean_delta_max=1e-4,
                loss_chunk=128,mlp_chunk=4096,checkpoint_group=3,alignment=128,
                optimizer_updates=0,training_eligible=False,automatic_expansion=False,
                scope='Single-turn training layouts up to 16K prefix/512 generation; not packed 64K or optimizer qualification')
    if getattr(args,'recurrent_state_precision',None) is not None:
        from ndm.recurrent_precision import validate_state_precision
        recipe['recurrent_state_precision']=validate_state_precision(args.recurrent_state_precision)
        recipe['scope']+='; explicit recurrent-state precision, unchanged weights and original behavior references'
    if getattr(args,'head_numeric_audit',False) and not getattr(args,'head_probe',False):
        raise ValueError('head numeric audit requires the read-only head probe')
    if getattr(args,'head_probe',False):
        recipe['head_probe']=dict(vocab_chunk=4096,max_rows=128,returns_original_outputs=True)
        recipe['scope']+='; read-only counterfactual FP32 head projection, not a changed-policy qualification'
        if getattr(args,'head_numeric_audit',False):
            recipe['head_probe']['numeric_audit']=True
            recipe['scope']+='; CPU FP64 selected dots and counterfactual BF16 re-quantization'
    args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
    publish(args.output/'recipe-private.json',recipe)
    print('RL_LOGPROB_ASSAY_FROZEN',len(selected),'turns',sha(args.output/'recipe-private.json'),flush=True)


def turn_layout(prefix, generated, device, alignment=128):
    import torch
    if not prefix or not generated or alignment<=0:
        raise ValueError('nonempty turn and positive alignment required')
    real=prefix+generated
    padded=((len(real)-2)//alignment+1)*alignment+1
    tokens=torch.tensor([real+[0]*(padded-len(real))],device=device,dtype=torch.long)
    valid=torch.zeros_like(tokens,dtype=torch.bool);valid[:,:len(real)]=True
    reset=torch.zeros_like(valid);reset[:,0]=True
    mask=torch.zeros((1,padded-1),device=device,dtype=torch.bool)
    mask[:,len(prefix)-1:len(real)-1]=True
    return tokens,valid,reset,mask


def run(args):
    import numpy as np
    import torch
    from ndm.e97 import load_e97_checkpoint,advance_e97_cache_segment,advance_e97_cache
    from scripts.qualify_e97_response_gradients import parameter_digest
    if sha(args.recipe)!=args.recipe_sha:raise ValueError('assay recipe identity')
    recipe=json.loads(args.recipe.read_text());target=recipe['model']
    if sha(target['checkpoint'])!=target['sha256'] or sha(recipe['args_json'])!=recipe['args_sha256']:
        raise ValueError('model identity')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single fixed numerical worker required')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=recipe['args_json'],device=torch.device('cuda',local),
                               dtype=torch.bfloat16,weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model.eval()
    from ndm.recurrent_precision import configure_recurrent_precision
    state_precision=configure_recurrent_precision(model,recipe.get('recurrent_state_precision'))
    if state_precision=='fp32' and (torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest'):
        raise ValueError('FP32 recurrent assay requires untruncated FP32 matrix arithmetic')
    before=parameter_digest(model)
    if any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('persistent parameter dtype')
    probes=[];probe_enabled='head_probe' in recipe
    if probe_enabled:
        from scripts.e97_head_precision_probe import HeadProbe
        base_probe=dict(vocab_chunk=4096,max_rows=128,returns_original_outputs=True)
        if recipe['head_probe'] not in (base_probe,dict(base_probe,numeric_audit=True)):
            raise ValueError('head probe recipe mismatch')
        if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':
            raise ValueError('head probe requires untruncated FP32 matrix arithmetic')
    records=[]
    with torch.no_grad():
        for item in recipe['selected']:
            probe=HeadProbe(item['generated'],numeric_audit=recipe['head_probe'].get('numeric_audit',False)) if probe_enabled else None
            handle=model.lm_head.register_forward_hook(probe.actor_hook) if probe is not None else None
            try:
                cache=advance_e97_cache_segment(loaded,item['prefix']);values=[]
                for token in item['generated']:
                    values.append(float(torch.log_softmax(cache.next_logits.float(),-1)[token].item()))
                    cache=advance_e97_cache(loaded,[token],cache)
            finally:
                if handle is not None:handle.remove()
            if probe is not None:probe.finish_actor();probes.append(probe)
            records.append(dict(id=item['id'],turn=item['turn'],actor_replay=values,recorded=item['recorded_logprobs']))
        del cache
        model.train();model.gradient_checkpointing=True;model.gradient_checkpoint_group_size=recipe['checkpoint_group']
        model.loss_chunk_size=recipe['loss_chunk'];model.loss_logits_fp32=True;model.checkpoint_loss_chunks=True
        modules=[m for m in model.modules() if hasattr(m,'checkpoint_chunk_size')]
        if len(modules)!=18:raise ValueError('expected 18 MLP modules')
        for m in modules:m.checkpoint_chunk_size=recipe['mlp_chunk']
        for index,(item,result) in enumerate(zip(recipe['selected'],records)):
            tokens,valid,reset,mask=turn_layout(item['prefix'],item['generated'],torch.device('cuda',local),recipe['alignment'])
            padded=tokens.shape[1]
            labels=tokens[:,1:];values=[];offset=0
            def head_hook(_module,_inputs,logits):
                nonlocal offset
                width=logits.shape[1];selected=mask[:,offset:offset+width]
                if bool(selected.any()):
                    target_ids=labels[:,offset:offset+width][selected]
                    lp=torch.log_softmax(logits[selected].float(),-1).gather(1,target_ids[:,None]).squeeze(1)
                    values.extend(lp.cpu().tolist())
                    if probe_enabled:
                        probes[index].observe_teacher(_module,_inputs[0][selected],logits[selected],target_ids)
                offset+=width
            handle=model.lm_head.register_forward_hook(head_hook)
            try:
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    loss=model(tokens,return_loss=True,loss_mask=mask,valid_mask=valid,
                               reset_before=reset,loss_reduction='sum')
            finally:handle.remove()
            if offset!=padded-1 or len(values)!=len(item['generated']):raise ValueError('teacher logprob/mask coverage')
            result.update(teacher=values,ce_mean=float(loss.item())/len(values),padded_tokens=padded)
            print('RL_LOGPROB_TURN_MEASURED',item['id'],item['turn'],flush=True)
    if probe_enabled:
        for result,probe in zip(records,probes):result['head_probe']=probe.report()
    after=parameter_digest(model)
    cached=np.array([abs(a-b) for r in records for a,b in zip(r['actor_replay'],r['recorded'])])
    teacher=np.array([abs(a-b) for r in records for a,b in zip(r['teacher'],r['recorded'])])
    ce=np.array([abs(r['ce_mean']+np.mean(r['teacher'])) for r in records])
    finite=all(np.isfinite(r[k]).all() for r in records for k in ('recorded','actor_replay','teacher','ce_mean'))
    checks=dict(finite=finite,parameters_unchanged=before==after,
                cached_replay=bool(cached.max()<=recipe['cached_replay_abs_max']),
                teacher_max=bool(teacher.max()<=recipe['teacher_abs_max']),
                teacher_p99=bool(np.quantile(teacher,.99)<=recipe['teacher_abs_p99']),
                ce_head_consistency=bool(ce.max()<=recipe['ce_mean_delta_max']))
    publish(args.output/'measurements-private.json',dict(records=records,recipe_sha256=args.recipe_sha))
    result=dict(status='measurements-complete',probability_path_passed=all(checks.values()),checks=checks,
                turns=len(records),tokens=len(cached),cached_replay_abs_max=float(cached.max()),
                teacher_abs_max=float(teacher.max()),teacher_abs_p99=float(np.quantile(teacher,.99)),
                ce_mean_delta_max=float(ce.max()),parameter_sha256=before,
                recipe_sha256=args.recipe_sha,peak_hbm_allocated=torch.cuda.max_memory_allocated(local),
                optimizer_updates=0,rl_optimizer_ready=False,automatic_expansion=False,
                remaining='End-to-end RL gradients, DDP/episode normalization, optimizer integration, reward contrast',scope=recipe['scope'])
    if 'recurrent_state_precision' in recipe:
        result['recurrent_state_precision']=state_precision
        if state_precision=='fp32':
            from ndm.recurrent_precision import FIXED_RECURRENT_KERNEL
            from ndm.triton.e88_triton_forward import _AUTOTUNE_CACHE
            pair=np.array([abs(a-b) for r in records for a,b in zip(r['teacher'],r['actor_replay'])])
            result['recurrent_kernel']=FIXED_RECURRENT_KERNEL
            result['legacy_autotune_cache_entries']=len(_AUTOTUNE_CACHE)
            result['matmul_policy']=dict(tf32=torch.backends.cuda.matmul.allow_tf32,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                float32_precision=torch.get_float32_matmul_precision())
            result['current_policy_pair']=dict(abs_max=float(pair.max()),abs_p99=float(np.quantile(pair,.99)),
                passed=bool(pair.max()<=recipe['teacher_abs_max'] and np.quantile(pair,.99)<=recipe['teacher_abs_p99']),
                scope='Actual current-policy forced-token scoring; NOT historical recorded behavior probabilities')
        if any(p.grad is not None for p in model.parameters()):raise ValueError('unexpected gradients in forward assay')
    if probe_enabled:
        pairs=[(a,b) for p in probes for a,b in zip(p.actor,p.teacher)]
        fp32=np.array([abs(a['fp32_logprob']-b['fp32_logprob']) for a,b in pairs])
        selected=np.array([abs(a['fp32_selected_logit']-b['fp32_selected_logit']) for a,b in pairs])
        bf16=np.array([abs(a['bf16_selected_logit']-b['bf16_selected_logit']) for a,b in pairs])
        if len(pairs)!=len(cached) or not all(np.isfinite(x).all() for x in (fp32,selected,bf16)):
            raise ValueError('head probe coverage/nonfinite')
        result['head_probe_diagnostic']=dict(tokens=len(pairs),fp32_pair_abs_max=float(fp32.max()),
            fp32_pair_abs_p99=float(np.quantile(fp32,.99)),fp32_selected_logit_abs_max=float(selected.max()),
            bf16_selected_logit_abs_max=float(bf16.max()),
            hidden_relative_l2_max=max(b['hidden_relative_l2'] for _,b in pairs),
            hidden_absolute_max=max(b['hidden_max_absolute'] for _,b in pairs),
            fp32_pair_within_original_limits=bool(fp32.max()<=recipe['teacher_abs_max'] and np.quantile(fp32,.99)<=recipe['teacher_abs_p99']),
            returns_original_outputs=True,changed_policy_qualified=False)
    publish(args.output/'summary.json',result);print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--output',type=Path,required=True);q.add_argument('--head-probe',action='store_true');q.add_argument('--head-numeric-audit',action='store_true');q.add_argument('--recurrent-state-precision',choices=('legacy','fp32'))
    q=s.add_parser('run');q.add_argument('--recipe',type=Path,required=True);q.add_argument('--recipe-sha',required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
