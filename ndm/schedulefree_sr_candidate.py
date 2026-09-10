"""Experimental BF16 Schedule-Free precision candidate; explicit SFT opt-in only.

Bounded parameter-device FP32 arithmetic, counter-based stochastic BF16 stores,
and exact BF16 live-y backup for eval/checkpoint basis transitions. No persistent
FP32 master weights. Eval/checkpoint adds one temporary host BF16 y copy.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import time
import torch
from ndm.schedulefree_offload import CPUOffloadAdamWScheduleFree

MASK = 0xffffffff
ALGORITHM = 'e97-sr-mix32-v1'


def mix32(x):
    # Multiplication stays below signed-int64 overflow for 32-bit inputs.
    x = (((x >> 16) ^ x) * 0x45d9f3b) & MASK
    x = (((x >> 16) ^ x) * 0x45d9f3b) & MASK
    return (x >> 16) ^ x


def counter_round(value, *, seed, step, parameter_id, stream, offset):
    if value.dtype != torch.float32:
        raise ValueError('rounding requires FP32 proposed value')
    if not bool(torch.isfinite(value).all()) or bool((value.abs() > torch.finfo(torch.bfloat16).max).any()):
        raise ValueError('nonfinite or overflowing proposed BF16 value')
    if not 0 <= offset <= MASK or offset + value.numel() > MASK + 1:
        raise ValueError('parameter coordinate exceeds counter range')
    key = mix32(seed) ^ mix32(step) ^ mix32(parameter_id) ^ mix32(stream)
    coordinates = torch.arange(offset, offset+value.numel(), device=value.device, dtype=torch.int64)
    noise = (mix32(coordinates ^ key) & 65535).to(torch.int32).reshape(value.shape)
    bits = value.contiguous().view(torch.int32)
    return ((bits + noise) >> 16).to(torch.int16).view(torch.bfloat16)


class ScheduleFreeSRCandidate(CPUOffloadAdamWScheduleFree):
    state_schema = 'emender-schedulefree-bf16-sr-candidate-v1'

    def __init__(self, named_parameters, *, seed=927413, bucket_numel=1048576, **kwargs):
        named = list(named_parameters)
        if not named or len({n for n,p in named}) != len(named) or len({id(p) for n,p in named}) != len(named):
            raise ValueError('unique named parameters required')
        if type(seed) is not int or not 0 <= seed <= MASK:
            raise ValueError('32-bit seed required')
        for name,p in named:
            if not isinstance(name,str) or not name or p.dtype != torch.bfloat16 or not p.is_contiguous() or not 0 < p.numel() <= MASK+1:
                raise ValueError('nonempty contiguous named BF16 parameters required')
        self.names = [n for n,p in named]
        self.ids = {p:int.from_bytes(hashlib.sha256(n.encode()).digest()[:4],'little') for n,p in named}
        if len(set(self.ids.values())) != len(named):
            raise ValueError('parameter counter-key collision')
        self.seed = seed
        self.merge_count = 0
        self.layout = hashlib.sha256(json.dumps([(n,list(p.shape)) for n,p in named],separators=(',',':')).encode()).hexdigest()
        self._eval_y = {}
        self._poisoned = False
        self._merge_active = False
        super().__init__([p for n,p in named], bucket_numel=bucket_numel, **kwargs)
        if not 0 < self.param_groups[0]['betas'][0] < 1:
            raise ValueError('candidate requires beta1 strictly between zero and one')
        for key in ('lr','eps','weight_decay','r','weight_lr_power'):
            if not math.isfinite(float(self.param_groups[0][key])):
                raise ValueError('finite optimizer hyperparameters required')

    def _identity(self):
        return {'schema':self.state_schema,'algorithm':ALGORITHM,'seed':self.seed,'layout_sha256':self.layout}

    def _healthy(self, *, allow_merge=False):
        if self._merge_active and not allow_merge:
            raise RuntimeError('outer merge must be committed before checkpoint/train/step')
        if self._poisoned:
            raise RuntimeError('partial failed operation: reload a committed checkpoint')
        if len(self.param_groups) != 1:
            raise ValueError('candidate supports exactly one parameter group')

    @torch.no_grad()
    def train(self):
        self._healthy()
        group = self.param_groups[0]
        if group['train_mode']:
            return
        if group['k'] > 0 and not self._eval_y:
            raise ValueError('saved/x checkpoint lacks exact live-y backup')
        self._poisoned = True
        devices=set()
        for p,y in self._eval_y.items():
            p.copy_(y,non_blocking=self.pin_memory and p.device.type=='cuda')
            if p.device.type=='cuda': devices.add(p.device)
        for device in devices: torch.cuda.synchronize(device)
        self._eval_y.clear()
        group['train_mode']=True
        self._poisoned=False

    @torch.no_grad()
    def eval(self):
        self._healthy()
        group=self.param_groups[0]
        if not group['train_mode']:
            return
        self._poisoned=True
        devices=set()
        for p in group['params']:
            self._eval_y[p]=self._new_host_tensor(p,source=p)
            if p.device.type=='cuda': devices.add(p.device)
            state=self.state.get(p,{})
            if 'z' not in state: continue
            for start in range(0,p.numel(),self.bucket_numel):
                sl=slice(start,start+self.bucket_numel)
                y=p.view(-1)[sl]
                z=state['z'].view(-1)[sl].to(p.device,non_blocking=self.pin_memory and p.device.type=='cuda')
                x=y.float().lerp(z.float(),1-1/group['betas'][0])
                if not bool(torch.isfinite(x).all()) or bool((x.abs()>torch.finfo(torch.bfloat16).max).any()):
                    raise ValueError('invalid averaged x export')
                y.copy_(x.to(torch.bfloat16))  # deterministic export; no SR counter consumed
        for device in devices: torch.cuda.synchronize(device)
        group['train_mode']=False
        self._poisoned=False

    @torch.no_grad()
    def store_merged_slice_(self, tensor, offset, value, label):
        """Bounded FP32 outer-average store; shared SR key on all replicas."""
        self._healthy(allow_merge=True)
        if self.param_groups[0]['train_mode'] or label not in {'sf_x','sf_z'}:
            raise ValueError('outer precision store requires x/z eval basis')
        slots={}
        for p in self.param_groups[0]['params']:
            target=p if label=='sf_x' else self.state[p]['z']
            slots[(target.untyped_storage().data_ptr(),target.storage_offset(),target.numel())]=p
        key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tensor.numel())
        if key not in slots:
            raise ValueError('outer store target is not an optimizer coordinate')
        self._merge_active=True
        try:
            rounded=counter_round(value,seed=self.seed,step=self.merge_count+1,
                parameter_id=self.ids[slots[key]],stream=0x6000b if label=='sf_x' else 0x7000d,offset=offset)
            tensor.view(-1)[offset:offset+value.numel()].copy_(rounded)
        except Exception:
            self._poisoned=True
            raise

    @torch.no_grad()
    def commit_merged_xz_(self):
        """Accept a real outer merge, unlike observational eval/train round trips.

        Caller has averaged x in model parameters and z in host state. Rebuild
        the new live y once, with a merge-specific counter shared by every rank.
        Local second moments are deliberately not merged or reset.
        """
        self._healthy(allow_merge=True)
        group=self.param_groups[0]
        if group['train_mode'] or set(self._eval_y)!=set(group['params']):
            raise ValueError('outer merge requires an explicit eval-basis export')
        if not 0<=self.merge_count<MASK:
            raise ValueError('merge counter exhausted')
        self._poisoned=True
        devices=set()
        for p in group['params']:
            state=self.state.get(p,{})
            if 'z' not in state:
                raise ValueError('outer merge requires initialized z for every parameter')
            if p.device.type=='cuda': devices.add(p.device)
            for start in range(0,p.numel(),self.bucket_numel):
                sl=slice(start,start+self.bucket_numel)
                x=p.view(-1)[sl]
                z=state['z'].view(-1)[sl].to(p.device).float()
                y=x.float().lerp(z,1-group['betas'][0])
                x.copy_(counter_round(y,seed=self.seed,step=self.merge_count+1,
                                      parameter_id=self.ids[p],stream=0x50009,offset=start))
        for device in devices: torch.cuda.synchronize(device)
        self._eval_y.clear()
        self.merge_count+=1
        self._merge_active=False
        group['train_mode']=True
        self._poisoned=False

    @torch.no_grad()
    def step(self, closure=None):
        self._healthy()
        group=self.param_groups[0]
        if not group['train_mode']:
            raise RuntimeError('call train() before step()')
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        k=int(group['k'])
        if not 0 <= k < MASK:
            raise ValueError('step exceeds SR counter range')
        if k == 0:
            # Make full-state checkpoint coverage unambiguous, including a
            # parameter that has not received a gradient in the first step.
            self.initialize_state_()
        self._poisoned=True
        begin=time.perf_counter()
        beta1,beta2=group['betas']
        warmup=group['warmup_steps']
        lr=float(group['lr'])*((k+1)/warmup if warmup and k<warmup else 1)
        group['scheduled_lr']=lr
        group['lr_max']=max(lr,group['lr_max'])
        weight=((k+1)**group['r'])*(group['lr_max']**group['weight_lr_power'])
        group['weight_sum']+=weight
        ckp1=weight/group['weight_sum'] if group['weight_sum'] else 0.0
        counts={'coordinates':0,'y_nonzero_proposals':0,'y_rne_would_stall':0,'y_sr_changed':0,'z_sr_changed':0}
        devices=set()
        for p in group['params']:
            if p.grad is None: continue
            if p.grad.is_sparse or not p.grad.is_contiguous():
                raise ValueError('dense contiguous gradient required')
            state=self._initialize_state(p)
            if p.device.type=='cuda': devices.add(p.device)
            for start in range(0,p.numel(),self.bucket_numel):
                sl=slice(start,start+self.bucket_numel)
                old=p.view(-1)[sl]
                # All optimizer arithmetic is on the parameter device, in FP32.
                y=old.float()
                z=state['z'].view(-1)[sl].to(p.device).float()
                v=state['exp_avg_sq'].view(-1)[sl].to(p.device).float()
                g=p.grad.view(-1)[sl].float()
                v.mul_(beta2).addcmul_(g,g,value=1-beta2)
                g.div_((v/(1-beta2**(k+1))).sqrt_().add_(group['eps']))
                if group['weight_decay']: g.add_(y,alpha=group['weight_decay'])
                proposed_y=y.lerp(z,ckp1).add_(g,alpha=lr*(beta1*(1-ckp1)-1))
                proposed_z=z.add(g,alpha=-lr)
                rounded=[]
                for stream,value in ((0x10001,proposed_y),(0x20003,proposed_z),(0x30007,v)):
                    rounded.append(counter_round(value,seed=self.seed,step=k+1,parameter_id=self.ids[p],stream=stream,offset=start))
                ry,rz,rv=rounded
                counts['coordinates']+=old.numel()
                nonzero=proposed_y!=y
                counts['y_nonzero_proposals']+=int(nonzero.sum())
                counts['y_rne_would_stall']+=int((nonzero & (proposed_y.bfloat16()==old)).sum())
                counts['y_sr_changed']+=int((ry!=old).sum())
                counts['z_sr_changed']+=int((rz!=z.bfloat16()).sum())
                old.copy_(ry)
                state['z'].view(-1)[sl].copy_(rz,non_blocking=self.pin_memory and p.device.type=='cuda')
                state['exp_avg_sq'].view(-1)[sl].copy_(rv,non_blocking=self.pin_memory and p.device.type=='cuda')
            if self.release_gradients: p.grad=None
        for device in devices: torch.cuda.synchronize(device)
        group['k']=k+1
        self.assert_state_offloaded()
        self._poisoned=False
        self.last_step_stats={**counts,'seconds':time.perf_counter()-begin,'bucket_numel':self.bucket_numel}
        return loss

    def state_dict(self):
        self._healthy()
        result=super().state_dict()
        result['precision_identity']=self._identity()
        result['rounding_counters']={'merge':self.merge_count}
        params=self.param_groups[0]['params']
        ids=result['param_groups'][0]['params']
        result['eval_live_y']={pid:self._eval_y[p] for pid,p in zip(ids,params) if p in self._eval_y}
        return result

    def load_state_dict(self, incoming):
        # Validate fully before delegating host-only z/v restoration; never retag v1.
        if set(incoming) != {'state','param_groups','precision_identity','eval_live_y','rounding_counters'} or incoming['precision_identity']!=self._identity():
            raise ValueError('SR checkpoint schema/seed/layout/algorithm mismatch')
        counters=incoming['rounding_counters']
        if not isinstance(counters,dict) or set(counters)!={'merge'} or type(counters['merge']) is not int or not 0<=counters['merge']<=MASK:
            raise ValueError('invalid merge rounding counter')
        groups=incoming['param_groups']
        if len(groups)!=1 or groups[0].get('state_schema')!=self.state_schema:
            raise ValueError('SR parameter-group schema mismatch')
        group=groups[0]; ids=group['params']; params=self.param_groups[0]['params']
        if len(ids)!=len(params) or len(set(ids))!=len(ids) or type(group['k']) is not int or not 0<=group['k']<=MASK:
            raise ValueError('SR counter or parameter mapping mismatch')
        backups=incoming['eval_live_y']
        if not isinstance(group['train_mode'],bool) or (group['train_mode'] and backups) or (not group['train_mode'] and group['k']>0 and set(backups)!=set(ids)):
            raise ValueError('missing/incompatible live-y checkpoint backup')
        if backups and set(backups)!=set(ids):
            raise ValueError('partial live-y backup')
        if set(backups)-set(ids) or set(incoming['state'])-set(ids):
            raise ValueError('unexpected optimizer slot')
        if group['k'] > 0 and (set(incoming['state']) != set(ids) or
                any(set(state) != {'z','exp_avg_sq'} for state in incoming['state'].values())):
            raise ValueError('partial initialized SR checkpoint state')
        for pid,p in zip(ids,params):
            tensors=list(incoming['state'].get(pid,{}).values())+([backups[pid]] if pid in backups else [])
            state=incoming['state'].get(pid,{})
            if state and set(state)!={'z','exp_avg_sq'}:
                raise ValueError('invalid SR tensor state')
            for tensor in tensors:
                if not torch.is_tensor(tensor) or tensor.dtype!=torch.bfloat16 or tensor.shape!=p.shape or tensor.device.type!='cpu' or not bool(torch.isfinite(tensor).all()):
                    raise ValueError('invalid BF16 host checkpoint tensor')
        for key in ('lr','eps','weight_decay','weight_sum','lr_max','r','weight_lr_power'):
            if not math.isfinite(float(group[key])): raise ValueError('nonfinite optimizer metadata')
        if not 0<group['betas'][0]<1 or not 0<=group['betas'][1]<1:
            raise ValueError('invalid restored betas')
        if group['lr']<0 or group['eps']<0 or group['weight_decay']<0 or group['weight_sum']<0 or type(group['warmup_steps']) is not int or group['warmup_steps']<0:
            raise ValueError('invalid restored hyperparameters')
        if any(bool((state['exp_avg_sq']<0).any()) for state in incoming['state'].values() if state):
            raise ValueError('negative second moment')
        self._poisoned=True
        super().load_state_dict({'state':incoming['state'],'param_groups':groups})
        self._eval_y={p:self._host_copy_for_parameter(p,backups[pid]) for pid,p in zip(ids,params) if pid in backups}
        self.merge_count=counters['merge']
        self._merge_active=False
        self._poisoned=False
