#!/usr/bin/env python3
"""Bounded, no-update E97 fused-recurrence and masked-opening gradient assay."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from ndm.e97 import load_e97_checkpoint
from ndm.e97_atomic import publish_bytes_no_replace
from ndm.triton.e97_sequential import e97_split_edit_triton_apply
from ndm.triton.e88_triton_forward import e88_torch_reference


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def publish(path, report):
    publish_bytes_no_replace(path,(json.dumps(report,indent=2,sort_keys=True)+'\n').encode(),mode=0o600)


def comparison(actual, reference, chunk=1048576):
    """Bounded-device gradient metrics; reference may reside on CPU."""
    assert actual.shape==reference.shape
    stats=torch.zeros(4,device=actual.device,dtype=torch.float64)
    a=actual.detach().reshape(-1); r=reference.detach().reshape(-1)
    for start in range(0,a.numel(),chunk):
        x=a[start:start+chunk].float(); y=r[start:start+chunk].to(a.device).float()
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise ValueError('nonfinite comparison')
        stats+=torch.stack(((x-y).double().square().sum(),y.double().square().sum(),
                            x.double().square().sum(),(x.double()*y.double()).sum()))
    error,rr,aa,dot=stats.tolist()
    return {'relative_l2':(error/rr)**0.5 if rr else (0.0 if error==0 else None),
            'reference_norm':rr**0.5,'actual_norm':aa**0.5,
            'cosine':dot/(rr*aa)**0.5 if rr and aa else (1.0 if rr==aa else 0.0)}


def require_close(stats, tolerance):
    if stats['relative_l2'] is None or stats['relative_l2']>tolerance:raise AssertionError(stats)


def aligned_length(length, alignment):
    if type(length) is not int or length<=0 or type(alignment) is not int or alignment<=0:
        raise ValueError('positive integer lengths and alignment required')
    return ((length+alignment-1)//alignment)*alignment


def recurrence_controls(length, device):
    padded=aligned_length(length,16)
    reset=torch.zeros(1,padded,device=device,dtype=torch.bool);reset[:,length//2]=True
    valid=torch.zeros_like(reset);valid[:,:length-3]=True
    return reset,valid


def kernel_assay(device, output, linear_state=False):
    cases=[]
    for length in (17,64):
        torch.manual_seed(97141+length)
        B,H,N=1,60,64
        padded=aligned_length(length,16)
        shape=(B,padded,H,N)
        base=[0.05*torch.randn(B,H,N,N,device=device),
              *[0.2*torch.randn(shape,device=device,dtype=torch.bfloat16) for _ in range(3)],
              (0.35+0.55*torch.rand(B,padded,H,device=device)).bfloat16(),
              0.2*torch.randn(shape,device=device,dtype=torch.bfloat16),
              torch.sigmoid(0.2*torch.randn(shape,device=device)).bfloat16(),
              torch.sigmoid(0.2*torch.randn(shape,device=device)).bfloat16()]
        tri=[t.detach().requires_grad_() for t in base]
        ref=[t.detach().float().clone().requires_grad_() for t in base]
        reset,valid=recurrence_controls(length,device)
        state,out=e97_split_edit_triton_apply(True,*tri[1:5],tri[5],tri[0],H,
            True,True,16,apply_silu_qkv=True,linear_state=linear_state,
            erase_gate=tri[6],value_write_gate=tri[7],reset_before=reset,valid_mask=valid)
        s,k,v,q,d,g,e,w=ref
        normalize=lambda x:F.normalize(F.silu(x),dim=-1,eps=1e-6)
        refout,refstate,_=e88_torch_reference(s,normalize(k).transpose(0,1),
            F.silu(v).transpose(0,1),normalize(q).transpose(0,1),d.transpose(0,1),
            linear_state=linear_state,erase_gate=e.transpose(0,1),value_write_gate=w.transpose(0,1),
            reset_before=reset.T.contiguous(),valid_mask=valid.T.contiguous())
        refout=refout.transpose(0,1)*F.silu(g)
        readout=torch.randn(out.shape,device=device)
        gt=torch.autograd.grad((out.float()*readout).sum()+0.1*state.square().sum(),tri)
        gr=torch.autograd.grad((refout*readout).sum()+0.1*refstate.square().sum(),ref)
        report={'length':length,'linear_state':linear_state,'padded_length':padded,'valid_length':length-3,'heads':H,'state':N,'projection_dtype':'bfloat16','reference':'float32 torch autograd',
                'output':comparison(out,refout),'final_state':comparison(state,refstate),
                'gradients':{name:comparison(a,b) for name,a,b in zip(('S0','k','v','q','decay','g','erase','write'),gt,gr)}}
        # Persist measurements before asserting: a failed assay remains inspectable.
        publish(output/f'kernel-{length}.json',report)
        require_close(report['output'],0.03);require_close(report['final_state'],0.03)
        for item in report['gradients'].values():require_close(item,0.03)
        assert torch.count_nonzero(out[:,length-3:])==0
        for gradient in gt[1:]:assert torch.count_nonzero(gradient[:,length-3:])==0
        cases.append(report)
        print('GRADIENT_KERNEL_PASS '+str(length),flush=True)
    return cases


def opening_inputs(example, device, alignment=1):
    prefix=example['prefix_tokens']; target=example['target_tokens']
    assert prefix and target and target[0]==32750
    real=prefix+target
    padded=aligned_length(len(real)-1,alignment)+1
    tokens=torch.tensor([real+[0]*(padded-len(real))],device=device,dtype=torch.long)
    valid=torch.zeros_like(tokens,dtype=torch.bool);valid[:,:len(real)]=True
    reset=torch.zeros_like(valid);reset[:,0]=True
    mask=torch.zeros((1,tokens.shape[1]-1),device=device,dtype=torch.bool)
    mask[:,len(prefix)-1]=True
    return tokens,valid,reset,mask


def parameter_digest(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(json.dumps([name,list(p.shape),str(p.dtype)]).encode())
        flat=p.detach().reshape(-1)
        for start in range(0,p.numel(),1048576):
            digest.update(flat[start:start+1048576].cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_assay(args,device,output):
    probes=json.loads(args.probes.read_text())
    assert probes['training_eligible'] is False
    example=next(x for x in probes['examples'] if x['id']=='consumed-agent-00000996')
    assert len(example['prefix_tokens'])==1102 and len(example['target_tokens'])==64
    loaded=load_e97_checkpoint(args.checkpoint,args_json=args.args_json,device=device,
        dtype=torch.bfloat16,weight_mode='train',use_triton=True,mmap=True)
    model=loaded.model.train()
    model.loss_logits_fp32=args.fp32_ce
    assert sum(p.numel() for p in model.parameters())==4045972080
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    tokens,valid,reset,mask=opening_inputs(example,device,alignment=16)
    head_name=next(name for name,p in model.named_parameters() if p is model.lm_head.weight)
    initial_parameters=parameter_digest(model)
    baseline=None; baseline_loss=None; reports=[]
    # First configuration is the dense CE/uncheckpointed reference. The second
    # copies the pilot's grouped checkpoint/MLP settings at a shorter length.
    cases=[('dense',False,0,0),('pilot-short',True,128,4096)]
    if args.diagnostic_chunks:
        cases.extend([('ce-only',True,73,4096),('mlp-only',True,128,256)])
    cases.append(('active-chunks',True,73,256))
    if args.diagnostic_chunks:cases.append(('pilot-repeat',True,128,4096))
    for label,gc_enabled,loss_chunk,mlp_chunk in cases:
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing=gc_enabled;model.gradient_checkpoint_group_size=3
        model.loss_chunk_size=loss_chunk
        model.checkpoint_loss_chunks=False  # matches the dense SFT trainer
        modules=[m for m in model.modules() if hasattr(m,'checkpoint_chunk_size')]
        assert len(modules)==18
        for module in modules:module.checkpoint_chunk_size=mlp_chunk
        embedding_checks=[]
        def capture_embedding(_module,_inputs,out):
            def capture_gradient(gradient):
                embedding_checks.append({'future_nonzero':int(torch.count_nonzero(gradient[:,len(example['prefix_tokens']):])),
                                         'prefix_nonzero':int(torch.count_nonzero(gradient[:,:len(example['prefix_tokens'])]))})
            out.register_hook(capture_gradient)
        hook=model.embedding.register_forward_hook(capture_embedding)
        head_observations=[];head_offset=0
        def capture_head(_module,_inputs,out):
            nonlocal head_offset
            position=len(example['prefix_tokens'])-1-head_offset
            if 0<=position<out.shape[1]:
                row=out[0,position].detach().float()
                head_observations.append({'dtype':str(out.dtype),
                    'cuda_autocast_enabled':torch.is_autocast_enabled('cuda'),
                    'target_nll_fp32':float(-row.log_softmax(-1)[32750]),
                    'top1_token':int(row.argmax()),'top1_probability':float(row.softmax(-1).max())})
            head_offset+=out.shape[1]
        head_hook=model.lm_head.register_forward_hook(capture_head)
        began=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(tokens,return_loss=True,loss_mask=mask,valid_mask=valid,reset_before=reset,loss_reduction='sum')
        loss.backward();hook.remove();head_hook.remove();torch.cuda.synchronize()
        assert torch.isfinite(loss) and len(embedding_checks)==1
        assert embedding_checks[0]['future_nonzero']==0 and embedding_checks[0]['prefix_nonzero']>0
        metrics={}
        if baseline is None:
            baseline={}
            for name,p in model.named_parameters():
                assert p.grad is not None,name
                metrics[name]=comparison(p.grad,p.grad)
                baseline[name]=p.grad.detach().cpu().clone()
            baseline_loss=float(loss.detach())
        else:
            for name,p in model.named_parameters():
                assert p.grad is not None,name
                metrics[name]=comparison(p.grad,baseline[name])
        report={'case':label,'example_id':example['id'],'prefix_tokens':1102,'target_token_count':1,
                'target_token':32750,'padded_input_steps':tokens.shape[1]-1,
                'valid_token_count':int(valid.sum()),'loss':float(loss.detach()),'loss_delta':float(loss.detach())-baseline_loss,
                'loss_dtype':str(loss.dtype),'head_observations':head_observations,
                'parameters':4045972080,'gradients':metrics,'embedding_gradient':embedding_checks[0],
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-began}
        assert len(head_observations)==1
        report['loss_logits_fp32']=args.fp32_ce
        report['ce_minus_fp32_nll']=report['loss']-head_observations[0]['target_nll_fp32']
        report['within_frozen_tolerances']=(abs(report['loss_delta'])<=0.05 and all(
            item['relative_l2'] is not None and item['relative_l2']<=0.05 for item in metrics.values()))
        publish(output/f'model-{label}.json',report)
        if args.fp32_ce:assert abs(report['ce_minus_fp32_nll'])<=1e-4
        if not args.diagnostic_chunks:
            assert report['within_frozen_tolerances']
        assert metrics[head_name]['actual_norm']>0
        reports.append(report)
        del loss
        print(('GRADIENT_MODEL_MEASURED ' if args.diagnostic_chunks else 'GRADIENT_MODEL_PASS ')+label,flush=True)
    final_parameters=parameter_digest(model)
    assert final_parameters==initial_parameters
    publish(output/'model-parameter-identity.json',{'before':initial_parameters,'after':final_parameters,'unchanged':True})
    return reports


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','args-json','probes','output-root'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('checkpoint-sha256','args-sha256','probes-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--diagnostic-chunks',action='store_true',help='measure failed chunk sensitivity without promoting the failed gate')
    p.add_argument('--fp32-ce',action='store_true',help='opt into temporary FP32 logits for CE; does not change parameters')
    args=p.parse_args()
    sources=[(args.checkpoint,args.checkpoint_sha256),(args.args_json,args.args_sha256),(args.probes,args.probes_sha256)]
    for path,expected in sources:assert sha(path)==expected,str(path)
    assert args.output_root.is_dir() and not list(args.output_root.iterdir())
    rank=int(os.environ['LOCAL_RANK']);assert 0<=rank<torch.cuda.device_count()
    torch.cuda.set_device(rank);assert torch.cuda.current_device()==rank
    device=torch.device('cuda',rank)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    print('GRADIENT_DEVICE '+json.dumps({'local_rank':rank,'current_device':torch.cuda.current_device(),
                                       'visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}),flush=True)
    kernel_assay(device,args.output_root)
    gc.collect();torch.cuda.empty_cache()
    reports=model_assay(args,device,args.output_root)
    for path,expected in sources:assert sha(path)==expected,str(path)
    report={'schema':'emender-e97-response-gradient-qualification-v1',
            'status':'completed-diagnostic' if args.diagnostic_chunks else 'passed',
            'qualification_pass':all(r['within_frozen_tolerances'] for r in reports),
            'loss_logits_fp32':args.fp32_ce,
            'ce_fp32_max_absolute_error':max(abs(r['ce_minus_fp32_nll']) for r in reports),
            'source_sha256':sha(__file__),'checkpoint_sha256':args.checkpoint_sha256,'weight_mode':'train',
            'torch':torch.__version__,'cuda':torch.version.cuda,'local_rank':rank,
            'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'gpu':torch.cuda.get_device_name(),
            'scope':'no updates; BF16 n64/H60 short recurrence reference and 4B single-opening gradient parity only; not 64K, full packed/DDP backward, effective optimizer updates, source admission, or capability success',
            'receipts':{path.name:sha(path) for path in sorted(args.output_root.iterdir())}}
    publish(args.output_root/'summary.json',report)
    print('RESPONSE_GRADIENT_QUALIFICATION_COMPLETE '+json.dumps(report),flush=True)


if __name__=='__main__':main()
