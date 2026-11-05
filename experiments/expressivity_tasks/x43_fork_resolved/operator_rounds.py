"""Inference-only bounded residual rounds. Never installs hooks or mutates weights.

k=1 is the original batched forward, preserving its floating point fingerprint.
For k>1, stores are local to a window, persistent across positions AND rounds.
B uses the independent SharedStore oracle; dense uses native continuation state.
Extra residual rounds occur at fork positions only. This intentionally slow
reference implementation is not a serving implementation.
"""
import math
import torch
from torch.nn import functional as F
from .c14_b_store import SharedStore
from .c14_b_suffix import queried_suffix


def fingerprint(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def weight_delta(before, after):
    """Slot strengths, with allocation/deletion counting as unit change."""
    return max((abs(a-b) for a,b in zip(before,after)), default=0.)


class IterativeForward:
    def __init__(self, model, *, store_arm, epsilon=1e-6):
        self.model=model
        self.store_arm=store_arm
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError('invalid epsilon')
        self.epsilon=epsilon
        self.telemetry=[]

    @torch.inference_mode()
    def __call__(self, ids, answers, contexts, *, windows, valid_lengths, forks, k):
        if self.model.training:
            raise ValueError('inference requires eval mode')
        if k not in (1,2,4):
            raise ValueError('k must be 1, 2, or 4')
        b,t=ids.shape
        if (answers.shape != (b,t,260) or contexts.shape != (b,t)
                or forks.shape != (b,t) or len(windows)!=b or len(valid_lengths)!=b
                or any(not 0 <= n <= t for n in valid_lengths)
                or answers.dtype != torch.float32 or not torch.isfinite(answers).all()):
            raise ValueError('invalid inputs or document window metadata')
        self.telemetry=[]
        # This compatibility path is deliberate: different GEMM shapes can
        # change bits even when the store equations agree mathematically.
        if k==1:
            if self.store_arm:
                return self.model(ids,answers,contexts,windows=windows,
                                  valid_lengths=valid_lengths,checkpointed=False)
            return self.model(ids,answers,contexts)
        base=self.model.head.base
        x=base.embed(ids)
        if not self.store_arm:
            x=x+self.model.device_input(answers)
        raw=[bytes(row) for row in ids.cpu().tolist()]
        for layer,block in enumerate(base.blocks):
            cell=block.cell
            rows=[]
            for j in range(b):
                states=[SharedStore(cfg) if self.store_arm else None for cfg in cell.configs]
                tokens=[]
                for i in range(t):
                    value=x[j:j+1,i:i+1]
                    if i>=valid_lengths[j]:
                        tokens.append(value)
                        continue
                    limit=k if bool(forks[j,i]) else 1
                    owner=queried_suffix(windows[j],raw[j][:i+1],int(contexts[j,i]))
                    for r in range(limit):
                        counts=0; delta=0.; reads=[]
                        with torch.autocast(device_type=ids.device.type,enabled=False):
                            stream=block.input(block.norm1(value)).float()
                            z=cell.W_in(stream)
                            vs=cell.W_v(stream).reshape(1,1,cell.policy.N,128)
                            av=cell.W_v(self.model.device_input(answers[j:j+1,i:i+1]).float()).reshape_as(vs)
                            pop=cell.W_pop(stream).sigmoid()
                            append=cell.W_append(stream); logits=cell.W_write(stream)
                            for h,cfg in enumerate(cell.configs):
                                features=z+cell.features[h](ids[j:j+1,i:i+1])
                                key=cell.geometry[h]['key'](features).detach()
                                query=cell.geometry[h]['query'](features)
                                if self.store_arm:
                                    s=states[h]
                                    before=[float(slot.strength) if slot else 0. for slot in s.slots]
                                    n=s.ops['CORRECT']
                                    content=vs[0,0,h]
                                    # Observe the current byte once; rounds must not
                                    # fabricate a new observed byte or shifted target.
                                    if r==0:
                                        s.observe(windows[j],raw[j][:i+1],raw[j][i],content,i)
                                    if bool(answers[j,i].abs().sum()) and owner.suffix:
                                        target,adjacent=s.target(i)
                                        prev=s.slots[target].value if adjacent else torch.zeros_like(content)
                                        a=av[0,0,h]
                                        err=(a-prev).square().sum().sqrt() if target is not None else a.new_zeros(())
                                        gate=(append[0,0,h]+cell.append_err_coef[h]*err).sigmoid()
                                        s.insert_array(key[0,0],a,owner,int(answers[j,i,:256].argmax()),i,gate)
                                    read,best,evidence=s.read(query[0,0],torch.zeros_like(content))
                                    s.stream(key[0,0],content,pop[0,0,h],append[0,0,h],logits[0,0,h],
                                             cell.append_err_coef[h],cell.match_coef[h],cell.error_coef[h],
                                             cell.gate_policies[h],best,evidence,i)
                                    s.decay()
                                    counts+=s.ops['CORRECT']-n
                                    after=[float(slot.strength) if slot else 0. for slot in s.slots]
                                    delta=max(delta,weight_delta(before,after))
                                    reads.append(read.reshape(1,1,128))
                                else:
                                    from ..x42_pile_scale.velocity_forward import tape_scan
                                    policy=cell.gate_policies[h]
                                    evidence=torch.empty(1,1,1,device=ids.device)
                                    old=states[h]
                                    if old is not None:
                                        # Conservative full-budget bound avoids one JIT
                                        # specialization per initial occupancy limit.
                                        old.live_bound=None
                                    before=None if old is None else old.native.strength.clone()
                                    read,state,trace,_=tape_scan(key.unsqueeze(2),query.unsqueeze(2),vs[:,:,h:h+1].contiguous(),
                                        torch.ones(1,1,1,device=ids.device),pop[:,:,h:h+1].contiguous(),append[:,:,h:h+1].contiguous(),
                                        old,cfg,cell.append_err_coef[h:h+1],wire_logits=logits[:,:,h:h+1].contiguous(),
                                        match_coef=cell.match_coef[h:h+1],error_coef=cell.error_coef[h:h+1],
                                        match_wire=policy.match_wire,occ_onset=policy.occ_onset,occ_slope=policy.occ_slope,evidence=evidence,error_evidence=torch.empty_like(evidence),
                                        occupancy=torch.empty_like(evidence),gate_values=torch.empty_like(evidence))
                                    states[h]=state
                                    # R-CONFLICT is the native residual correction route.
                                    from ..x30_tape_cell.routing_trace import CASE_NAMES
                                    counts+=sum(int((trace.case==case).sum()) for case,name in CASE_NAMES.items() if name == 'R-CONFLICT')
                                    delta=max(delta,1. if before is None else float((before-state.native.strength).abs().max()))
                                    reads.append(read.flatten(2))
                        value=value+block.read_out(torch.cat(reads,-1))
                        value=value+block.ff(block.norm2(value))
                        halt=counts==0 and delta<self.epsilon
                        self.telemetry.append(dict(layer=layer,batch=j,position=i,round=r+1,
                            fork=bool(forks[j,i]),CORRECT=counts,weight_delta=delta,halt=halt,
                            branch_reads=1))
                        if halt: break
                    tokens.append(value)
                rows.append(torch.cat(tokens,1))
            x=torch.cat(rows,0)
        lp=F.linear(base.norm(x).float(),base.head.weight.float()).log_softmax(-1)
        return lp,torch.ones_like(contexts)
