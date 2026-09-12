"""Read-only head observers. Counterfactual FP32 outputs never replace model outputs."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def project(head, hidden, logits, targets, vocab_chunk=4096):
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
        return [dict(fp32_logprob=p,fp32_selected_logit=f,bf16_selected_logit=b)
                for p,f,b in zip(fp32_lp.cpu().tolist(),fp32_selected.cpu().tolist(),bf16_selected.cpu().tolist())]


class HeadProbe:
    def __init__(self,generated):
        self.generated=generated;self.calls=0;self.hidden=[];self.actor=[];self.teacher=[]

    def actor_hook(self,head,inputs,logits):
        i=self.calls;self.calls+=1
        if i>=len(self.generated):return None  # final consumed token's prediction is unused
        h=inputs[0][:,-1,:]
        if h.shape[0]!=1:raise ValueError('single actor head row required')
        target=torch.tensor([self.generated[i]],device=h.device,dtype=torch.long)
        self.actor.extend(project(head,h,logits[:,-1,:],target))
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
        rows=project(head,hidden,logits,targets)
        for row,r,a in zip(rows,relative.cpu().tolist(),absolute.cpu().tolist()):
            row.update(hidden_relative_l2=r,hidden_max_absolute=a)
        self.teacher.extend(rows)

    def report(self):
        self.finish_actor()
        if len(self.teacher)!=len(self.generated):raise ValueError('teacher head hook coverage')
        return dict(actor=self.actor,teacher=self.teacher)
