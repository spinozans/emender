#!/usr/bin/env python3
"""Bound first-mixer component observations to the complete-matrix reproduction."""
import argparse
import json
import os
from pathlib import Path
from scripts.diagnose_e97_activation_alignment import ROOT,PROFILES,KEYS,evaluate,compare_banks
from scripts.diagnose_e97_observer_effect import delta
from scripts.e97_first_mixer_probe import FirstMixerProbe
from scripts.eval_e97_native_execution import publish,sha

REFERENCE=ROOT/'native-activation-alignment-reproduction-v3'
REFERENCE_SHA='8336d6d5c21b4b4746a44d9e15d7b8978f5a910b3a25ea511257a21c862d02fa'
SUMMARY_SHA='04f42be946e9226152e18d18bde252a1a451062745a728e91da642ec49866594'
RECIPE_SHA='1aebe6302d42032ca5f9a0af3d63cc5f36d7f401b509801d1a483952cb23d9d5'
NAMES=('actor-cache','full-eval','masked-padded-eval','training-default')


def freeze(args):
    for name,digest in (('measurements-private.json',REFERENCE_SHA),('summary.json',SUMMARY_SHA),('recipe-private.json',RECIPE_SHA)):
        if sha(REFERENCE/name)!=digest:raise ValueError('reference identity')
    if not json.loads((REFERENCE/'summary.json').read_text())['localization_ready']:raise ValueError('unbound reference')
    source=json.loads((REFERENCE/'recipe-private.json').read_text());refs={(r['id'],r['turn']):r for r in json.loads((REFERENCE/'measurements-private.json').read_text())['reports']}
    selected=[]
    for item in source['selected']:
        if not 0<len(item['prefix'])<=4096 or not 0<len(item['generated'])<=512:raise ValueError('component input bounds')
        length=len(item['prefix'])+len(item['generated'])-1
        if length*(12*3840+60)*2>512*1024**2:raise ValueError('component bank preflight')
        selected.append(dict(item,reference_logprobs={p['profile']['name']:p['logprobs'] for p in refs[(item['id'],item['turn'])]['profiles'] if p['profile']['name'] in NAMES}))
    recipe=dict(schema='emender-e97-first-mixer-v1',selected=selected,profiles=[p for p in PROFILES if p['name'] in NAMES],
                model=source['model'],args_json=source['args_json'],args_sha256=source['args_sha256'],parameter_sha256=source['parameter_sha256'],
                reference_sha256=REFERENCE_SHA,reference_summary_sha256=SUMMARY_SHA,source_recipe_sha256=RECIPE_SHA,
                repetitions=2,workers=1,limit=1e-4,max_bank_bytes=512*1024**2,cpu_trace_limit_bytes=3*1024**3,
                optimizer_updates=0,training_eligible=False,automatic_expansion=False,
                scope='32 evaluations; entire real layer-0 prefix, not merely prediction rows. Read-only module hooks; no numerical fix or RL qualification')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False);publish(args.output/'recipe-private.json',recipe)
    print('FIRST_MIXER_FROZEN',sha(args.output/'recipe-private.json'),flush=True)


def observed(loaded,item,profile,budget):
    probe=FirstMixerProbe(loaded.model,len(item['prefix'])+len(item['generated'])-1,budget)
    try:
        coarse,lp=evaluate(loaded,item,profile);del coarse
        bank=probe.finish();return bank,lp,probe.chunk_lengths
    finally:probe.close()


def run(args):
    import torch
    from ndm.e97 import load_e97_checkpoint
    from scripts.audit_e97_live_actor_capture import fingerprint,runtime
    if sha(args.recipe)!=args.recipe_sha:raise ValueError('recipe identity')
    recipe=json.loads(args.recipe.read_text());target=recipe['model'];local=int(os.environ.get('LOCAL_RANK','0'))
    if recipe['schema']!='emender-e97-first-mixer-v1' or [(r['id'],r['turn']) for r in recipe['selected']]!=KEYS:raise ValueError('selection identity')
    if recipe['max_bank_bytes']!=512*1024**2 or recipe['cpu_trace_limit_bytes']!=3*1024**3:raise ValueError('storage limits')
    if recipe['profiles']!=[p for p in PROFILES if p['name'] in NAMES] or recipe['repetitions']!=2 or recipe['limit']!=1e-4:raise ValueError('execution recipe')
    if recipe['workers']!=1 or int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single worker required')
    for path,digest in ((target['checkpoint'],target['sha256']),(recipe['args_json'],recipe['args_sha256'])):
        if sha(path)!=digest:raise ValueError('model identity')
    torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':raise ValueError('precision defaults')
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=recipe['args_json'],device=torch.device('cuda',local),dtype=torch.bfloat16,weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model;before=fingerprint(model);m=model.layers[0].mixer
    if before['parameters']!=recipe['parameter_sha256'] or any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('effective model identity')
    if len(model.layers)!=18 or any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules()):raise ValueError('deterministic model required')
    if m.key_dim!=3840 or m.value_dim!=3840 or m.n_heads!=60 or m.projection_chunk_size!=512 or m.head_mix!='concat' or m.use_conv:raise ValueError('component model shape')
    publish(args.output/'preflight.json',dict(recipe_sha256=args.recipe_sha,fingerprint=before,runtime=runtime(loaded,local),projection_chunk_size=m.projection_chunk_size))
    private=args.output/'profiles-private';private.mkdir(mode=0o700,exist_ok=False);rows=[];receipts={}
    for item in recipe['selected']:
        actor=None
        if not 0<len(item['prefix'])<=4096 or not 0<len(item['generated'])<=512:raise ValueError('input bounds')
        for profile in recipe['profiles']:
            torch.manual_seed(974223);bank,lp,chunks=observed(loaded,item,profile,recipe['max_bank_bytes'])
            if sum(v.numel()*v.element_size() for v in bank.values())*6>recipe['cpu_trace_limit_bytes']:raise ValueError('trace budget')
            torch.manual_seed(974223);repeated,repeated_lp,repeated_chunks=observed(loaded,item,profile,recipe['max_bank_bytes'])
            repeat=compare_banks(bank,repeated);del repeated
            if actor is None:actor=bank
            comparison=compare_banks(actor,bank)
            row=dict(id=item['id'],turn=item['turn'],profile=profile,logprobs=lp,projection_chunk_lengths=chunks,
                     reference_max_delta=delta(lp,item['reference_logprobs'][profile['name']]),repeat_logprob_max_delta=delta(lp,repeated_lp),
                     repeat_equal=chunks==repeated_chunks and all(r['reference_sha256']==r['current_sha256'] and not r['dtype_changed'] for r in repeat),
                     against_actor=comparison,repeat_comparison=repeat,
                     first_differing_site=next((r['stage'] for r in comparison if r['differing_elements'] or r['dtype_changed']),None))
            path=private/f"{item['id']}-turn-{item['turn']}-{profile['name']}.json"
            publish(path,dict(recipe_sha256=args.recipe_sha,measurement=row));receipts[path.name]=sha(path);rows.append(row)
            print('FIRST_MIXER_PROFILE_COMPLETE',item['id'],item['turn'],profile['name'],row['reference_max_delta'],row['first_differing_site'],flush=True)
    after=fingerprint(model)
    if before!=after or any(p.grad is not None for p in model.parameters()):raise ValueError('state/gradient mutation')
    publish(args.output/'measurements-private.json',dict(recipe_sha256=args.recipe_sha,rows=rows))
    bound=all(r['reference_max_delta']<=recipe['limit'] for r in rows);repeat=all(r['repeat_equal'] and r['repeat_logprob_max_delta']<=recipe['limit'] for r in rows)
    publish(args.output/'summary.json',dict(status='diagnostic-measurements-complete',recipe_sha256=args.recipe_sha,receipts=receipts,fingerprint=before,
        reference_binding_passed=bound,repeatability_passed=repeat,component_localization_ready=bound and repeat,
        profiles=[{k:v for k,v in r.items() if k not in ('logprobs','against_actor','repeat_comparison')} for r in rows],
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,rl_optimizer_ready=False,original_probability_gate='failed, unchanged',peak_hbm_allocated=torch.cuda.max_memory_allocated(local)))
    print('FIRST_MIXER_COMPLETE',sha(args.output/'summary.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('run');q.add_argument('--output',type=Path,required=True);q.add_argument('--recipe',type=Path,required=True);q.add_argument('--recipe-sha',required=True)
    a=p.parse_args();globals()[a.command](a)
