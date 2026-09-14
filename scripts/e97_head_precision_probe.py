"""Read-only head observers. Counterfactual FP32 outputs never replace model outputs."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def project(head, hidden, logits, targets, vocab_chunk=4096, *, numeric_audit=False):
    if hidden.ndim!=2 or not 0<len(hidden)<=128 or not 0<vocab_chunk<=4096:
        raise ValueError('bounded head probe shape')
    if head.weight.dtype!=torch.bfloat16 or hidden.dtype!=torch.bfloat16 or logits.dtype!=torch.bfloat16:
        raise ValueError('expected original BF16 projection')
    if logits.shape!=(len(hidden),head.weight.shape[0]) or targets.shape!=(len(hidden),) or targets.dtype!=torch.long:
        raise ValueError('head probe target coverage')
    if bool(((targets<0)|(targets>=logits.shape[-1])).any()):raise ValueError('head probe target range')
    # At most 128 x vocabulary FP32 outputs plus a 4096 x hidden-width
    # temporary FP32 weight block. No persistent FP32 parameter copy.
    with torch.autocast(device_type=hidden.device.type,enabled=False):
        h=hidden.float();output=torch.empty(logits.shape,device=hidden.device,dtype=torch.float32)
        for start in range(0,output.shape[1],vocab_chunk):
            stop=min(start+vocab_chunk,output.shape[1])
            bias=None if head.bias is None else head.bias[start:stop].float()
            output[:,start:stop]=F.linear(h,head.weight[start:stop].float(),bias)
        fp32_selected=output.gather(1,targets[:,None]).squeeze(1)
        fp32_lp=torch.log_softmax(output,-1).gather(1,targets[:,None]).squeeze(1)
        bf16_selected=logits.float().gather(1,targets[:,None]).squeeze(1)
        if not all(bool(torch.isfinite(x).all()) for x in (output,fp32_lp,bf16_selected)):
            raise ValueError('nonfinite head probe')
        rows=[dict(fp32_logprob=p,fp32_selected_logit=f,bf16_selected_logit=b)
              for p,f,b in zip(fp32_lp.cpu().tolist(),fp32_selected.cpu().tolist(),bf16_selected.cpu().tolist())]
        if numeric_audit:
            # Counterfactual re-quantization: does the large gap return when the
            # FP32 GEMM output is stored in BF16? Never return these as model logits.
            rounded=output.bfloat16().float()
            rounded_lp=torch.log_softmax(rounded,-1).gather(1,targets[:,None]).squeeze(1)
            native_lp=torch.log_softmax(logits.float(),-1).gather(1,targets[:,None]).squeeze(1)
            rounding_disagreement=(rounded-logits.float()).abs().amax(dim=1)
            # Independent CPU FP64 selected dot products of the actual BF16
            # operands. Only <=128 selected weight rows, not an FP64 model copy.
            selected_weight=head.weight.detach().index_select(0,targets).cpu().double()
            selected64=(hidden.detach().cpu().double()*selected_weight).sum(dim=1)
            if head.bias is not None:selected64+=head.bias.detach().index_select(0,targets).cpu().double()
            if not all(bool(torch.isfinite(t).all()) for t in (selected64,rounded_lp,native_lp,rounding_disagreement)):
                raise ValueError('nonfinite head numeric audit')
            for row,d,r,native,error in zip(rows,selected64.tolist(),rounded_lp.cpu().tolist(),native_lp.cpu().tolist(),rounding_disagreement.cpu().tolist()):
                row.update(fp64_selected_logit=d,rounded_fp32_logprob=r,native_logprob=native,
                    native_vs_rounded_fp32_logit_abs_max=error,
                    fp32_vs_fp64_selected_abs=abs(row['fp32_selected_logit']-d))
        return rows


class HeadProbe:
    def __init__(self,generated,*,numeric_audit=False):
        self.numeric_audit=numeric_audit
        self.generated=generated;self.calls=0;self.hidden=[];self.actor=[];self.teacher=[]

    def actor_hook(self,head,inputs,logits):
        i=self.calls;self.calls+=1
        if i>=len(self.generated):return None  # final consumed token's prediction is unused
        h=inputs[0][:,-1,:]
        if h.shape[0]!=1:raise ValueError('single actor head row required')
        target=torch.tensor([self.generated[i]],device=h.device,dtype=torch.long)
        self.actor.extend(project(head,h,logits[:,-1,:],target,numeric_audit=self.numeric_audit))
        self.hidden.append(h.detach().cpu().clone())
        return None

    def finish_actor(self):
        if self.calls!=len(self.generated)+1 or len(self.actor)!=len(self.generated):
            raise ValueError('actor head hook coverage')

    def observe_teacher(self,head,hidden,logits,targets):
        start=len(self.teacher);stop=start+len(hidden)
        if targets.cpu().tolist()!=self.generated[start:stop]:raise ValueError('teacher head target alignment')
        reference=torch.cat(self.hidden[start:stop],dim=0).to(hidden.device).float()
        delta=hidden.float()-reference
        relative=delta.norm(dim=1)/reference.norm(dim=1).clamp_min(1e-30)
        absolute=delta.abs().amax(dim=1)
        if not bool(torch.isfinite(relative).all() & torch.isfinite(absolute).all()):
            raise ValueError('nonfinite hidden comparison')
        rows=project(head,hidden,logits,targets,numeric_audit=self.numeric_audit)
        for row,r,a in zip(rows,relative.cpu().tolist(),absolute.cpu().tolist()):
            row.update(hidden_relative_l2=r,hidden_max_absolute=a)
        self.teacher.extend(rows)

    def report(self):
        self.finish_actor()
        if len(self.teacher)!=len(self.generated):raise ValueError('teacher head hook coverage')
        return dict(actor=self.actor,teacher=self.teacher)
