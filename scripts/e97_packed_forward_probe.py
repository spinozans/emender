"""Pure layout and passive head-observer helpers for the packed64K assay."""
import hashlib
import math
import torch


def digest(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(str((str(t.dtype),list(t.shape))).encode()+t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def assemble(records,order,context=65536,pad=0):
    if not order or len(set(order))!=len(order):raise ValueError('unique whole records required')
    tokens=torch.full((context+1,),pad,dtype=torch.long);tm=torch.zeros(context+1,dtype=torch.bool)
    valid=torch.zeros_like(tm);reset=torch.zeros_like(tm);starts={};pos=0
    for rid in order:
        rec=records[rid];n=len(rec['tokens'])
        if n<2 or rec['mask'].shape!=rec['tokens'].shape or pos+n>context+1:raise ValueError('record/layout bounds')
        starts[rid]=pos;tokens[pos:pos+n]=rec['tokens'];tm[pos:pos+n]=rec['mask'];valid[pos:pos+n]=True;reset[pos]=True;pos+=n
    loss=tm[1:] & valid[:-1] & valid[1:] & ~reset[1:]
    return dict(tokens=tokens,mask=loss,valid=valid,reset=reset,starts=starts,length=pos,order=list(order))


def probes(record):
    n=len(record['tokens']);mask=record['mask']
    # First eight prompt predictions are diagnostic, NOT supervised targets.
    for start in range(8,n-32):
        if bool(mask[start+1:start+33].all()):return dict(boundary=list(range(8)),assistant=list(range(start,start+32)))
    raise ValueError('record lacks a 32-token assistant span')


def rotate(order,k):return order[k:]+order[:k]


def variants(records,order,sentinels,context=65536,vocab=50281):
    anchor=sentinels[0]
    if anchor!=order[0]:raise ValueError('anchor must be first authority record')
    base=assemble(records,order,context)
    # Choose offsets by token geometry alone, before seeing any model output.
    late=assemble(records,rotate(order,1),context);last=late['starts'][anchor]
    candidates=list(range(1,len(order)))
    if context==65536:
        candidates=[j for j in candidates if all(
            sum(len(records[x]['tokens']) for x in order[j:])%mod not in (0,last%mod) for mod in (16,512))]
    if not candidates:raise ValueError('no rotation meets placement residues')
    k=min(candidates,key=lambda j:abs(sum(len(records[x]['tokens']) for x in order[j:])-context//2))
    middle=assemble(records,rotate(order,k),context)
    positions=[x['starts'][anchor] for x in (base,middle,late)]
    if context==65536 and (not 28000<=positions[1]<=38000 or positions[2]<55000
            or len({x%512 for x in positions})<3 or len({x%16 for x in positions})<3):raise ValueError('placement coverage')
    changed={k:(v.clone() if torch.is_tensor(v) else v) for k,v in late.items()}
    start=late['starts'][anchor];changed['tokens'][:start]=(changed['tokens'][:start]+1)%vocab
    noreset={k:(v.clone() if torch.is_tensor(v) else v) for k,v in changed.items()};noreset['reset'][start]=False
    padding={k:(v.clone() if torch.is_tensor(v) else v) for k,v in base.items()}
    if base['length']>=context:raise ValueError('tail padding required')
    padding['tokens'][base['length']:]=(padding['tokens'][base['length']:]+1)%vocab
    return dict(original=base,middle=middle,late=late,predecessor_changed=changed,reset_removed=noreset,padding_changed=padding)


class HeadRows:
    """Observe actual FP32 logits; never replace an output or change a mask."""
    def __init__(self,batch,rows):
        self.batch=batch;self.rows=rows;self.offset=0;self.parts={k:[] for k in rows};self.loss_values=[]
    def hook(self,module,inputs,logits):
        if logits.dtype!=torch.float32 or inputs[0].dtype!=torch.bfloat16:raise ValueError('head/storage dtype')
        if not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite actual logits')
        width=logits.shape[1];lo=self.offset;hi=lo+width
        labels=self.batch['tokens'][lo+1:hi+1]
        lp=torch.log_softmax(logits[0],-1).gather(1,labels[:,None]).squeeze(1)
        mask=self.batch['mask'][lo:hi];self.loss_values.extend(lp[mask].cpu().tolist())
        for name,positions in self.rows.items():
            local=[p-lo for p in positions if lo<=p<hi]
            if local:
                idx=torch.tensor(local,device=logits.device)
                self.parts[name].append((lp[idx].cpu(),inputs[0][0,idx].detach().cpu(),logits[0,idx].detach().cpu()))
        self.offset=hi
        return None
    def finish(self,loss):
        if self.offset!=len(self.batch['mask']) or len(self.loss_values)!=int(self.batch['mask'].sum()):raise ValueError('head/loss coverage')
        if not self.loss_values or not math.isfinite(loss):raise ValueError('loss finite/count')
        result={}
        for name,parts in self.parts.items():
            if not parts:raise ValueError('missing probe')
            lp,h,z=[torch.cat([p[i] for p in parts]) for i in range(3)]
            if len(lp)!=len(self.rows[name]):raise ValueError('probe coverage')
            result[name]=dict(logprobs=lp.tolist(),hidden_sha256=digest(h),logits_sha256=digest(z))
        return dict(probes=result,loss_sum=loss,targets=len(self.loss_values),
                    ce_delta=abs(loss+math.fsum(self.loss_values))/len(self.loss_values))


def state_hashes(states):
    def tensors(x):
        if torch.is_tensor(x):return [x]
        if isinstance(x,(tuple,list)):return [t for y in x for t in tensors(y)]
        if x is None:return []
        raise ValueError('unexpected recurrent state structure')
    if len(states)!=18:raise ValueError('expected18 layer states')
    hashes=[]
    for layer in states:
        values=tensors(layer)
        if not values or any(t.dtype!=torch.float32 for t in values):raise ValueError('FP32 carry required')
        flat=torch.cat([t.reshape(-1) for t in values])
        if flat.numel()!=60*64*64 or not bool(torch.isfinite(flat).all()):raise ValueError('state shape/nonfinite')
        hashes.append(digest(flat))
    return hashes
