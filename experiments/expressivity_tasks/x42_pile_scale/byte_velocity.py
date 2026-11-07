"""Independent byte-native CMA-ES services; token search sources stay unchanged."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np
import torch
from .byte_stream import ByteStream, digest
from .cmaes import CMAES
from .velocity_state import atomic_json, save_generation, restore_generation
from .velocity_candidate import Candidate, Policy
from .velocity_parent import DIMENSIONS as PURE_DIMENSIONS
from .velocity_baseline_search import dimensions as baseline_dimensions, SearchPolicy
from .velocity_score import score_metrics
from .velocity_train import configure_production, telemetry

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
REG = ROOT/'byte_velocity_registration.json'
PYTHON = '/home/erikg/emender/.venv/bin/python'
CARDS = dict(transformer='2', gdn2='3', pure='4')
CAPACITY = 150_003_984


def dimensions(arm):
    original = PURE_DIMENSIONS if arm == 'pure' else baseline_dimensions(arm)
    return [(name, 1e-5 if name == 'lr' else 10. if name == 'warmup_steps' else lo,
             3e-3 if name == 'lr' else 8000. if name == 'warmup_steps' else hi, center, log)
            for name, lo, hi, center, log in original]


def decode(arm, vector):
    space = dimensions(arm)
    if len(vector) != len(space) or not np.isfinite(vector).all():
        raise ValueError('invalid vector')
    values = {}
    for u, (name, lo, hi, _, log) in zip(1-np.abs(np.mod(vector, 2)-1), space):
        value = min(hi, max(lo, math.exp(math.log(lo)+u*math.log(hi/lo)) if log else lo+u*(hi-lo)))
        values[name] = round(value) if name in ('N','budget_min','w_in_width','warmup_steps') else value
    if arm == 'pure':
        for name in ('match_wire','occ_onset','occ_slope'):
            values[name] = tuple(values.pop(f'{name}{h}') for h in range(8))
        return Policy(**values)
    values['microbatch'] = 2**round(values.pop('microbatch_log2'))
    if arm == 'gdn2':
        values['expansion'] = (1., 2.)[round(values.pop('expansion_index'))]
    return SearchPolicy(**values)


def verify():
    if sys.executable != PYTHON:
        raise RuntimeError('approved Python required')
    reg = json.loads(REG.read_text())
    if subprocess.check_output(['git','show','HEAD:'+str(REG.relative_to(REPO))], cwd=REPO) != REG.read_bytes():
        raise ValueError('registration must be committed')
    for path, expected in reg['sources'].items():
        if digest(REPO/path) != expected:
            raise ValueError('changed source: '+path)
    for arm in CARDS:
        if reg['dimensions'][arm] != [list(d) for d in dimensions(arm)]:
            raise ValueError('dimension mismatch')
    stream = reg['stream']
    for key in ('offsets', 'documents_index', 'binding_receipt'):
        if digest(stream[key]) != stream[key+'_sha256']:
            raise ValueError('stream identity mismatch: '+key)
    st = Path(stream['source']).stat()
    if [st.st_size, st.st_ino, st.st_mtime_ns] != stream['source_stat']:
        raise ValueError('source identity changed')
    import importlib.metadata
    if importlib.metadata.version('flash-linear-attention') != reg['fla_version']:
        raise ValueError('FLA identity changed')
    return reg


def model_for(arm, policy):
    if arm == 'pure':
        with torch.device('meta'):
            probe = Candidate(policy, vocab=256, target=CAPACITY)
        frozen = probe.counts()['total']-probe.counts()['trainable']
        del probe
        model = Candidate(policy, vocab=256, target=CAPACITY-frozen)
    else:
        from . import velocity_baselines as base
        from ..x40_emender_lm import scale_surface, part8_surface
        from ndm.models import external_gdn2
        original = external_gdn2.GDN2ExternalMLPLayer
        def expanded(*args, **kwargs):
            return original(*args, **{**kwargs, 'expansion': policy.expansion})
        with patch.object(base, 'VOCAB', 256), patch.object(external_gdn2, 'GDN2ExternalMLPLayer', expanded):
            model = base.Baseline(arm)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    module.weight.mul_(policy.init_scale)
            if arm == 'gdn2':
                for block in model.model.blocks:
                    core = block.layer.gdn2.gdn2
                    core.A_log.add_(math.log(policy.decay_rate_scale))
                    core.dt_bias.add_(policy.dt_bias_shift)
    total = sum(p.numel() for p in model.parameters())
    if abs(total/CAPACITY-1) >= .001:
        raise ValueError('total capacity mismatch: '+str(total))
    return model


def grouped(rows, microbatch):
    # Equal-length subgroups need no padding; each row remains a separate reset.
    for at in range(0, len(rows), microbatch):
        chunk = rows[at:at+microbatch]
        for length in sorted(set(len(r[2]) for r in chunk)):
            selected = [r for r in chunk if len(r[2]) == length]
            yield (torch.as_tensor(np.stack([r[2] for r in selected]), device='cuda'),
                   torch.as_tensor(np.stack([r[3] for r in selected]), device='cuda'))


def evaluate(model, stream, arm):
    model.eval()
    losses = []; samples = []; targets = 0
    with torch.no_grad():
        for i, row in enumerate(stream.windows('valid')):
            if i == 24:
                break
            x, y = next(grouped([row], 1))
            losses.append(float(model.loss(x,y))*y.numel()); targets += y.numel()
            if arm == 'pure':
                samples.append(telemetry(model))
        if not targets:
            raise ValueError('empty heldout')
        nll = sum(losses)/targets
        if arm != 'pure':
            return dict(score=nll, nll_heldout=nll, eval_targets=targets)
        peaks = []
        for length in (512,1024,2048,4096):
            row = next(r for r in stream.windows('valid', length) if len(r[2]) == length)
            x,y = next(grouped([row],1)); model.loss(x,y)
            peaks.append([r['peak_live_per_head'][0] for r in telemetry(model)])
        count = len(samples[0])
        slopes = [0. if min(p[h] for p in peaks)<=0 else float(np.polyfit(np.log([512,1024,2048,4096]),np.log([p[h] for p in peaks]),1)[0]) for h in range(count)]
        result = score_metrics(nll_heldout=nll,
            occ_frac=float(np.mean([s[h]['occ_frac'] for s in samples for h in range(count)])),
            ever_touched_fractions=[float(np.mean([s[h]['ever_touched_fraction'] for s in samples])) for h in range(count)],
            r_cap_fire_rate=float(np.mean([s[h]['rcap_fraction_per_head'][0] for s in samples for h in range(count)]))*model.policy.N,
            occupancy_slope=max(slopes))
        return dict(**result, eval_targets=targets, occupancy_slopes=slopes)


def train(arm, policy_path, output):
    reg = verify()
    card = CARDS[arm]
    if os.environ.get('GPU_LEASE_HELD') != card or os.environ.get('CUDA_VISIBLE_DEVICES') != card:
        raise ValueError('leased device mismatch')
    torch.set_num_threads(1); torch.manual_seed(3401); torch.cuda.manual_seed_all(3401)
    configure_production()
    cls = Policy if arm == 'pure' else SearchPolicy
    policy = cls(**json.loads(Path(policy_path).read_text()))
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    model = model_for(arm, policy).cuda().train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=policy.lr,
        betas=(.9, .95 if arm == 'pure' else policy.beta2), weight_decay=.1 if arm == 'pure' else policy.weight_decay, fused=True)
    stream = ByteStream(reg['stream']['source'], reg['stream']['offsets'])
    atomic_json(out/'start.json', dict(arm=arm, policy=asdict(policy), registration_sha256=digest(REG),
        stream_identity=reg['stream'], physical_gpu=card, pid=os.getpid(), parameters=dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)), seed=3401))
    batches = iter(stream.batches('train'))
    elapsed=0.; step=targets=0
    with (out/'training.jsonl').open('w', buffering=1) as log:
        while elapsed < 2700:
            torch.cuda.synchronize(); start=time.monotonic()
            rows = next(batches)  # exhaustion is a failure, never wrap
            n = sum(len(r[3]) for r in rows)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = policy.lr*min(1., (step+1)/policy.warmup_steps)
            loss_sum = 0.
            for x,y in grouped(rows, 8 if arm == 'pure' else policy.microbatch):
                loss = model.loss(x,y)
                if arm == 'pure':
                    loss = loss + policy.gate_cost_lambda*model.gate_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                (loss*(y.numel()/n)).backward(); loss_sum += float(loss.detach())*y.numel()/n
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            torch.cuda.synchronize(); elapsed += time.monotonic()-start; step+=1; targets+=n
            log.write(json.dumps(dict(step=step, active_seconds=elapsed, bytes=targets, loss=loss_sum, time_ns=time.time_ns()))+'\n')
    os.environ['X40_BATCHED_READS']='0'
    result=evaluate(model,stream,arm)
    result.update(status='completed',score_eligible=True,arm=arm,train_seconds=elapsed,steps=step,bytes=targets,units='nats/byte',registration_sha256=digest(REG))
    atomic_json(out/'result.json',result)


def parent(arm):
    verify()
    out=ROOT/'runs'/('byte_velocity_'+arm+'_v1'); out.mkdir(parents=True,exist_ok=True)
    active=out/'active.json'
    if active.exists():
        saved=json.loads(active.read_text()); proc=Path(f"/proc/{saved['pid']}/stat")
        if proc.exists() and proc.read_text().split()[21] == saved['start_ticks']:
            raise RuntimeError('previous candidate still alive; refusing duplicate launch')
    state=out/'generation.json'; identity=digest(REG)
    if state.exists():
        es,completed=restore_generation(state,identity)
    else:
        center=[math.log(c/lo)/math.log(hi/lo) if log else (c-lo)/(hi-lo) for _,lo,hi,c,log in dimensions(arm)]
        es=CMAES(center, population=12, seed=4301); completed=[]
        es.ask(); save_generation(state,es,identity,completed)
    while es.generation < 12:
        for i in range(len(completed),12):
            candidate=out/f'g{es.generation+1:02d}_c{i:02d}'; candidate.mkdir(exist_ok=True)
            policy=candidate/'policy.json'; atomic_json(policy,asdict(decode(arm,es.x[i])))
            attempt=1
            while (candidate/f'attempt{attempt}').exists(): attempt+=1
            output=candidate/f'attempt{attempt}'
            env=dict(os.environ,GPU_LEASE_VISIBLE=CARDS[arm],GPU_LEASE_HB_INTERVAL='30',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTORCH_ALLOC_CONF='expandable_segments:True')
            with (candidate/f'console{attempt}.txt').open('w') as log:
                child=subprocess.Popen(['bash',str(ROOT/'run_byte_velocity_leased.sh'),arm,str(policy),str(output)],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(out/'active.json',dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],output=str(output)))
            status=child.wait()
            if status:
                raise RuntimeError(f'candidate failure exit={status}: {output}')
            result=json.loads((output/'result.json').read_text())
            if result['status']!='completed' or not result['score_eligible'] or result['train_seconds']<2700 or result['registration_sha256']!=identity:
                raise ValueError('ineligible result')
            completed.append(result['score']); save_generation(state,es,identity,completed)
        es.tell(completed); completed=[]
        if es.generation<12: es.ask()
        save_generation(state,es,identity,completed)
    results=[json.loads(p.read_text())|dict(path=str(p)) for p in out.glob('g*/attempt*/result.json')]
    winner=min(results,key=lambda r:r['score']); atomic_json(out/'nomination.json',winner)


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('parent','train')); p.add_argument('--arm',choices=CARDS,required=True); p.add_argument('--policy'); p.add_argument('--output'); a=p.parse_args()
    try:
        if a.mode=='parent': parent(a.arm)
        else: train(a.arm,a.policy,a.output)
    except BaseException:
        out=Path(a.output).parent if a.output else ROOT/'runs'/('byte_velocity_'+a.arm+'_v1')
        atomic_json(out/'failure.json',dict(status='failed',time_ns=time.time_ns(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__': main()
