#!/usr/bin/env python3
"""Capture real first-chunk operands, bind a minimal replay, then isolate two controls."""
import argparse
import json
import os
from pathlib import Path
from scripts.eval_e97_native_execution import publish,sha
from scripts.diagnose_e97_first_divergence import reference,evaluate,exact_scores
from scripts.e97_first_divergence import digest
from scripts.e97_tensor_capsule import unpack
from scripts.e97_recurrent_call_capture import Capture,describe

TRACE=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/first-divergence-v1')
TRACE_SHA='34c7b1dd33ef245b3c5c63b99071a0b6912d38cbed2088537ebc5631dcc88c4a'
MASKS=('reset_before','valid_mask')


def freeze(args):
    reference()
    if sha(TRACE/'summary.json')!=TRACE_SHA:raise ValueError('trace reference')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    publish(args.output/'plan.json',dict(schema='e97-recurrent-operands-v1',trace_summary_sha256=TRACE_SHA,
        key=['onpolicy-task-004-sample-0',1],layer=0,chunk=0,geometry=[512,1,60,64],repetitions=2,
        full_model_evaluations=4,maximum_micro_evaluations=12,sequential_workers=2,
        controls=['captured actor vs teacher masks','captured actor vs teacher discarded-workspace policy'],
        matrix_condition='identical numeric values/dtypes/strides/aliases, common other flags, semantically equivalent masks on zero initial state',
        capsule_byte_limit=256*1024**2,max_capture_hbm=16*1024**3,max_micro_hbm=2*1024**3,
        capture_seconds=900,replay_seconds=300,outer_seconds=1500,teardown_seconds=30,
        native_binding='exact scores, CE, fine trace hashes and kernel output/readout binding',
        micro_binding='exact captured output and final-state bytes on both original controls; exact repeats',
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,
        scope='Read-only real-operand localization; matrix uses fixed allocations for input operands, not output/workspace addresses; not64K/optimizer qualification'))
    print('RECURRENT_OPERANDS_FROZEN',sha(args.output/'plan.json'),flush=True)


def device():
    import torch
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single worker required')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    return torch.device('cuda',local)


def write_capsule(path,value):
    import torch
    with path.open('xb') as f:torch.save(value,f)
    path.chmod(0o400)


def capture(args):
    import torch
    from ndm.e97 import load_e97_checkpoint
    from ndm.numerical_policy import configure_numerical_policy
    from scripts.audit_e97_live_actor_capture import fingerprint
    from ndm.triton.e88_triton_forward import _AUTOTUNE_CACHE
    plan=json.loads((args.output/'plan.json').read_text());dev=device();base,item,ref=reference()
    if sha(TRACE/'summary.json')!=TRACE_SHA:raise ValueError('trace reference')
    target=base['model']
    for p,h in ((target['checkpoint'],target['sha256']),(base['args_json'],base['args_sha256'])):
        if sha(p)!=h:raise ValueError('model identity')
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=base['args_json'],device=dev,dtype=torch.bfloat16,
        weight_mode=target['mode'],use_triton=True,mmap=True)
    configure_numerical_policy(loaded.model,base['numerical_policy']);before=fingerprint(loaded.model)
    if before['parameters']!='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd':raise ValueError('effective parameters')
    hashes={}
    for profile in ('actor','teacher'):
        prior=json.loads((TRACE/f'fine-{profile}-0-private.json').read_text());first=None
        for repeat in (0,1):
            probe=Capture()
            try:bank,receipt=evaluate(loaded,item,base,profile,0)
            finally:probe.close()
            evidence=probe.finish();outputs=unpack(probe.outputs,'cpu');inputs=unpack(probe.inputs,'cpu')
            if list(inputs['k'].shape)!=plan['geometry'] or inputs['S0'].dtype!=torch.float32:raise ValueError('captured geometry/state')
            bound=exact_scores(receipt['scores'],ref['actor_replay' if profile=='actor' else 'teacher'])
            if profile=='teacher':bound=bound and receipt['ce_mean']==ref['ce_mean'] and receipt['padded_tokens']==ref['padded_tokens']
            trace_bound={k:digest(v) for k,v in bank.items()}==prior['tensor_sha256']
            readout_bound=digest(outputs['out'].transpose(0,1).reshape(512,3840))==digest(bank['o_proj.input'][:512])
            evidence.update(native_bound=bound,trace_bound=trace_bound,readout_bound=readout_bound)
            name=f'{profile}-{repeat}-capsule-private.pt'
            write_capsule(args.output/name,dict(inputs=probe.inputs,outputs=probe.outputs,meta=probe.meta,plan_sha256=args.plan_sha))
            hashes[name]=sha(args.output/name)
            publish(args.output/f'{profile}-{repeat}-receipt-private.json',dict(evidence=evidence,native=receipt))
            comparable={k:evidence[k] for k in ('inputs','outputs','meta')}
            if first is None:first=comparable
            elif comparable!=first:raise ValueError('capture repeat differs')
            if not (bound and trace_bound and readout_bound):raise ValueError('capture not reference-bound')
            del bank,inputs,outputs,probe
            print('RECURRENT_CAPTURE_BOUND',profile,repeat,flush=True)
    after=fingerprint(loaded.model);peak=torch.cuda.max_memory_allocated(dev)
    if before!=after or _AUTOTUNE_CACHE or any(p.grad is not None for p in loaded.model.parameters()):raise ValueError('model/tuner mutation')
    if peak>plan['max_capture_hbm']:raise ValueError('capture HBM limit')
    publish(args.output/'capture-summary.json',dict(capture_ready=True,hashes=hashes,parameter_fingerprint=after,
        peak_hbm_allocated=peak,plan_sha256=args.plan_sha,optimizer_updates=0))


def delta(a,b):
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('output geometry/dtype')
    bits=(a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8)).reshape(*a.shape,a.element_size()).any(-1)
    locations=bits.nonzero()
    return dict(exact=not bool(bits.any()),differing_elements=int(bits.sum()),
        first_index=locations[0].tolist() if len(locations) else None,absolute_max=float((a.float()-b.float()).abs().max()))


def eligible(a,b,ma,mb):
    import torch
    numeric=[k for k in a if k not in MASKS]
    values=all((a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and digest(a[k].cpu())==digest(b[k].cpu())) for k in numeric)
    layouts=all(a[k] is None or (b[k] is not None and a[k].stride()==b[k].stride() and a[k].storage_offset()==b[k].storage_offset()) for k in numeric)
    aliases=lambda x:{(u,v) for u in numeric for v in numeric if x[u] is not None and x[v] is not None and x[u].untyped_storage().data_ptr()==x[v].untyped_storage().data_ptr()}
    common={k:v for k,v in ma.items() if k!='recurrent_state_precision'}=={k:v for k,v in mb.items() if k!='recurrent_state_precision'}
    masks=True
    for x in (a,b):
        masks=masks and bool((x['S0']==0).all()) and not bool(torch.signbit(x['S0']).any())
        masks=masks and (x['valid_mask'] is None or bool(x['valid_mask'].all()))
        masks=masks and (x['reset_before'] is None or not bool(x['reset_before'][1:].any()))
    return dict(numeric_values_equal=values,numeric_layouts_equal=layouts,aliases_equal=aliases(a)==aliases(b),
        other_flags_equal=common,masks_equivalent_on_zero_state=masks,
        workspace_modes_are_original_pair={ma['recurrent_state_precision'],mb['recurrent_state_precision']}=={'legacy','fp32'})


def replay(args):
    import torch
    from ndm.triton.e88_triton_backward import e88_triton
    from ndm.triton.e88_triton_forward import _AUTOTUNE_CACHE
    plan=json.loads((args.output/'plan.json').read_text());dev=device()
    if sha(args.output/'capture-summary.json')!=args.capture_sha:raise ValueError('capture summary identity')
    capture_summary=json.loads((args.output/'capture-summary.json').read_text())
    if not capture_summary['capture_ready'] or capture_summary['plan_sha256']!=args.plan_sha:raise ValueError('capture not ready')
    capsules={};inputs={};expected={};baselines={}
    for profile in ('actor','teacher'):
        name=f'{profile}-0-capsule-private.pt'
        if sha(args.output/name)!=capture_summary['hashes'][name]:raise ValueError('capsule identity')
        c=torch.load(args.output/name,map_location='cpu',weights_only=True);capsules[profile]=c
        inputs[profile]=unpack(c['inputs'],dev);expected[profile]=unpack(c['outputs'],'cpu')
        if c['plan_sha256']!=args.plan_sha:raise ValueError('capsule plan')
    def execute(tensors,meta):
        with torch.no_grad():out,state=e88_triton(**tensors,**meta)
        result=dict(out=out.cpu(),S_final=state.cpu())
        if not all(bool(torch.isfinite(v).all()) for v in result.values()):raise ValueError('nonfinite micro output')
        return result
    for profile in ('actor','teacher'):
        rows=[]
        for repeat in (0,1):
            result=execute(inputs[profile],capsules[profile]['meta'])
            row={k:delta(expected[profile][k],result[k]) for k in result};rows.append(row)
            publish(args.output/f'replay-{profile}-{repeat}.json',row)
            if not all(x['exact'] for x in row.values()):raise ValueError('minimal replay does not reproduce captured output/state')
        baselines[profile]=rows
    print('RECURRENT_MICRO_BASELINES_BOUND',flush=True)
    checks=eligible(inputs['actor'],inputs['teacher'],capsules['actor']['meta'],capsules['teacher']['meta'])
    publish(args.output/'matrix-preflight.json',dict(checks=checks,meta={p:c['meta'] for p,c in capsules.items()},
        numeric_comparison={k:delta(inputs['actor'][k].cpu(),inputs['teacher'][k].cpu()) for k in inputs['actor'] if k not in MASKS and inputs['actor'][k] is not None and inputs['teacher'][k] is not None and inputs['actor'][k].shape==inputs['teacher'][k].shape}))
    matrix=[]
    if all(checks.values()):
        for mask_profile,workspace_profile in (('actor','actor'),('teacher','actor'),('actor','teacher'),('teacher','teacher')):
            tensors=dict(inputs['actor']);tensors.update({k:inputs[mask_profile][k] for k in MASKS})
            meta=dict(capsules['actor']['meta']);meta['recurrent_state_precision']=capsules[workspace_profile]['meta']['recurrent_state_precision']
            name=f'mask-{mask_profile}-workspace-{workspace_profile}';prior=None
            for repeat in (0,1):
                result=execute(tensors,meta);hashes={k:digest(v) for k,v in result.items()}
                if prior is None:prior=hashes;write_capsule(args.output/f'{name}-outputs-private.pt',result)
                elif hashes!=prior:raise ValueError('matrix repeat differs')
            row=dict(name=name,hashes=prior,input_addresses={k:v.data_ptr() if v is not None else None for k,v in tensors.items()},against_actor={k:delta(expected['actor'][k],result[k]) for k in result},
                against_teacher={k:delta(expected['teacher'][k],result[k]) for k in result},repeat_exact=True)
            publish(args.output/f'{name}.json',row);matrix.append(row)
            print('RECURRENT_CONTROL_MEASURED',name,flush=True)
    if _AUTOTUNE_CACHE or torch.cuda.max_memory_allocated(dev)>plan['max_micro_hbm']:raise ValueError('micro resource/tuner guard')
    # Kernel inputs are required to remain immutable throughout every intervention.
    for p in inputs:
        original=unpack(capsules[p]['inputs'],'cpu')
        if any(digest(v.cpu() if v is not None else None)!=digest(original[k]) for k,v in inputs[p].items()):raise ValueError('micro mutated inputs')
    publish(args.output/'replay-summary.json',dict(micro_reference_bound=True,matrix_executed=bool(matrix),checks=checks,
        matrix=matrix,peak_hbm_allocated=torch.cuda.max_memory_allocated(dev),optimizer_updates=0,
        numerical_gate_still_failed=True,training_eligible=False,
        caveat='Workspace intervention changes allocation and compiled storage code together; not a separation of those mechanisms.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','capture','replay'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan-sha');p.add_argument('--capture-sha');a=p.parse_args()
    try:
        if a.command!='freeze' and sha(a.output/'plan.json')!=a.plan_sha:raise ValueError('plan identity')
        globals()[a.command](a)
    except Exception as e:
        if a.output.exists():publish(a.output/f'{a.command}-failure.json',dict(error_type=type(e).__name__,message=str(e),localization_ready=False))
        raise
