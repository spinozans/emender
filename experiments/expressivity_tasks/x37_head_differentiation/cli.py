"""Sequential broker-only mechanism experiment; raw receipts, resumable checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
from .model import LM, ARMS, parameter_receipt, input_snapshot, input_metrics
from ..x34_lm_pilot import data
from ..x34_lm_pilot.cli import write, save_checkpoint, replay_present, validate
from ..x34_lm_pilot.model import LM as Control
from ..x34_lm_pilot.probes import story_probe, depth_probe, context_probe, aggregate_depth
from ..x36_tape_perf.steady import FiniteGradients

ROOT = Path(__file__).resolve().parent
REC = ROOT/'receipts'
STEPS = 3052
MILESTONES = (0,64,512,1536,STEPS)


def sources():
    paths = list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    paths += [ROOT/'RECEIPTS.md',ROOT/'run_leased.sh']
    paths += list((ROOT.parent/'x31_tape_cell_fused').glob('*.py'))
    for folder,names in (
        ('x30_tape_cell',('naive_reference','routing_trace')),
        ('x34_lm_pilot',('model','data','probes','cli')),
        ('x36_tape_perf',('cell_fused','forward','backward','kernels','journal',
                            'loss','replay_setup','mutations','steady'))):
        paths += [ROOT.parent/folder/(name+'.py') for name in names]
    return {str(p.relative_to(ROOT.parent)):data.sha(p) for p in sorted(paths)}


def primary(stories):
    peaks = [r['telemetry'][0]['peak_live_per_head'] for r in stories]
    assert len(peaks) == 16 and all(len(p)==4 for p in peaks)
    return dict(identical_stories=sum(len(set(p))==1 for p in peaks),stories=16,
                fraction_identical=sum(len(set(p))==1 for p in peaks)/16,
                mean_peak_per_head=np.mean(peaks,axis=0).tolist(),peaks=peaks)


def audit():
    receipt = ROOT.parent/'x34_lm_pilot/receipts/probe_pilot_tape_lm.json'
    control = json.loads(receipt.read_text())
    torch.manual_seed(3401)
    model = Control('tape_lm')
    cell = model.blocks[2].cell
    rows = {}
    for name in ('W_k','W_q','W_v','W_write','W_pop','W_append'):
        w = getattr(cell,name).weight.detach().reshape(4,-1)
        rows[name] = {f'0-{h}':float((w[0]-w[h]).norm()) for h in (1,2,3)}
    del model
    result = dict(control_receipt=str(receipt.relative_to(ROOT.parent)),
                  control_receipt_sha256=data.sha(receipt),
                  tokens=control['tokens'],primary=primary(control['stories']),
                  initial_head_weight_distances=rows,
                  conclusion='Equal occupancy does not imply identical head parameters; '
                  'X34 already has unequal head rows at initialization. '
                  'F12g age-1 correction may enforce identical occupancy without '
                  'content-independent read dynamics.',
                  x36_summary_sha256=data.sha(ROOT.parent/'x36_tape_perf/receipts/summary.json'))
    assert result['primary']['identical_stories'] == 16
    assert all(v>0 for group in rows.values() for v in group.values())
    write(REC/'control.json',result)
    return result


def gate():
    for name,minimum in (('oracle.xml',141),('mechanism.xml',4)):
        path = REC/name
        suites = ET.parse(path).getroot()
        cases = suites.findall('.//testcase')
        if len(cases)<minimum or any(c.find('failure') is not None or
             c.find('error') is not None or c.find('skipped') is not None for c in cases):
            raise RuntimeError(f'full passing pretraining gate required: {name}')
    expected = json.loads((REC/'gate_sources.json').read_text())
    if expected != sources():
        raise RuntimeError('source/preregistration changed after parity gates')


def evaluate(model, tok, step, initial):
    tic = time.perf_counter()
    model.eval()
    stories = story_probe(model,tok,data.RUNS/'valid.txt','cuda',stories=16)
    result = dict(step=step,tokens=step*32768,stories=stories,primary=primary(stories),
                  validation=validate(model,'cuda'),
                  W_in=input_metrics(model.blocks[2].cell,initial))
    result['seconds'] = time.perf_counter()-tic
    model.train()
    return result


def fingerprint(arm):
    return dict(arm=arm,seed=3401,batch=64,context=512,steps=STEPS,
                sources=sources(),tokenizer=data.sha(data.RUNS/'tokenizer.json'),
                data_receipt=data.sha(data.ROOT/'receipts/data.json'),
                train_bin=data.sha(data.RUNS/'train.bin'),
                valid_bin=data.sha(data.RUNS/'valid.bin'))


def train(arm,tok):
    gate()
    torch.manual_seed(3401);np.random.seed(3401);random.seed(3401)
    model = LM(arm,tok.get_vocab_size())
    write(REC/f'parameters_{arm}.json',parameter_receipt(model,tok.get_vocab_size()))
    model.cuda().train()
    initial = input_snapshot(model.blocks[2].cell)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    run = ROOT/'runs'/arm
    run.mkdir(parents=True,exist_ok=True)
    path = run/'latest.pt'
    fp = fingerprint(arm)
    step,elapsed = 0,0.
    if path.exists():
        saved = torch.load(path,map_location='cuda',weights_only=False)
        if fp != saved['fingerprint']:
            raise RuntimeError('resume fingerprint mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        initial = saved['initial'];step = saved['step'];elapsed = saved['elapsed_seconds']
        torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        del saved
        if (run/'curve.jsonl').exists():
            rows = [json.loads(line) for line in (run/'curve.jsonl').read_text().splitlines()]
            abandoned = [r for r in rows if r['step']>step]
            if abandoned:
                write(run/f'abandoned_{time.time_ns()}.json',abandoned)
                (run/'curve.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=step))
    write(run/'preregistration.json',dict(fingerprint=fp,sha256=data.sha(ROOT/'RECEIPTS.md')))
    started = time.perf_counter()
    finite = FiniteGradients('cuda')
    status = 'failed'
    pending = []

    def checkpoint():
        save_checkpoint(path,dict(fingerprint=fp,step=step,model=model.state_dict(),
            optimizer=optimizer.state_dict(),initial=initial,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),elapsed_seconds=elapsed+time.perf_counter()-started))

    def flush(log):
        nonlocal pending
        torch.cuda.synchronize()
        finite.check()
        for number,loss,norm,start,end,host_seconds in pending:
            value = float(loss)
            if not math.isfinite(value):
                raise FloatingPointError('nonfinite training loss')
            log.write(json.dumps(dict(step=number,tokens=number*32768,loss=value,
                grad_norm=float(norm),gpu_step_ms=start.elapsed_time(end),
                host_dispatch_seconds=host_seconds,replay_node='ReplayBackward'))+'\n')
        pending = []

    try:
        if step == 0 and not (REC/f'eval_{arm}_0.json').exists():
            write(REC/f'eval_{arm}_0.json',evaluate(model,tok,0,initial))
        with open(run/'curve.jsonl','a',buffering=1) as log:
            while step<STEPS:
                tic = time.perf_counter()
                x,y = (a.pin_memory().cuda(non_blocking=True)
                       for a in data.batch('train',step,batch_size=64))
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                optimizer.zero_grad(set_to_none=True)
                logits,_ = model(x)
                loss = F.cross_entropy(logits.flatten(0,1),y.flatten())
                replay_present(loss)
                loss.backward()
                norm = finite.clip(model.parameters())
                optimizer.step();end.record();step += 1
                pending.append((step,loss.detach(),norm.detach(),start,end,time.perf_counter()-tic))
                del logits,loss
                if step%64 == 0 or step == STEPS:
                    flush(log)
                    if step in MILESTONES:
                        write(REC/f'eval_{arm}_{step}.json',evaluate(model,tok,step,initial))
                    checkpoint()
                    print(json.dumps(dict(arm=arm,step=step,tokens=step*32768,
                        wall_seconds=elapsed+time.perf_counter()-started)),flush=True)
        status = 'completed'
    except BaseException as exc:
        write(REC/f'failure_{arm}_{time.time_ns()}.json',dict(step=step,error=repr(exc)))
        raise
    finally:
        # Save only completed/verified boundaries. Exceptions remain explicit and
        # never become a resumable success checkpoint.
        rows = [json.loads(line) for line in (run/'curve.jsonl').read_text().splitlines()] if (run/'curve.jsonl').exists() else []
        write(REC/f'train_{arm}.json',dict(status=status,steps=step,tokens=step*32768,
            elapsed_seconds=elapsed+time.perf_counter()-started,physical_gpu=os.environ['GPU_LEASE_HELD'],
            torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
            samples=len(rows),median_gpu_step_ms=statistics.median(r['gpu_step_ms'] for r in rows[2:]) if len(rows)>2 else None,
            checkpoint_step=rows[-1]['step'] if status=='completed' and rows else None,
            fingerprint=fp))
    return model


def winner_probe(model,arm,tok):
    model.eval()
    valid = np.memmap(data.RUNS/'valid.bin',mode='r',dtype=np.uint16)
    result = dict(arm=arm,stories=story_probe(model,tok,data.RUNS/'valid.txt','cuda'),
        depth=depth_probe(model,tok,'cuda'),context=context_probe(model,valid,'cuda'))
    result['depth_bins'] = aggregate_depth(result['depth'])
    write(REC/'winner_probe.json',result)


def summarize():
    control = json.loads((REC/'control.json').read_text())
    x34 = json.loads((data.ROOT/'receipts/pilot_tape_lm.json').read_text())
    reference = x34['curve'][-1]['validation']
    arms = {}
    for arm in ARMS:
        final = json.loads((REC/f'eval_{arm}_{STEPS}.json').read_text())
        init = json.loads((REC/f'eval_{arm}_0.json').read_text())
        ratio = final['validation']['perplexity']/reference['perplexity']
        history = [json.loads((REC/f'eval_{arm}_{i}.json').read_text()) for i in MILESTONES]
        distances = [r['W_in']['distance_from_init'] for r in history]
        monotonic = [all(b[h]>=a[h] for a,b in zip(distances,distances[1:]))
                     for h in range(len(distances[0]))]
        arms[arm] = dict(primary=final['primary'],initial_primary=init['primary'],
            validation=final['validation'],perplexity_ratio_to_control=ratio,
            quality_within_2pct=abs(ratio-1)<=.02,quality_over_5pct_worse=ratio>1.05,
            W_in=final['W_in'],W_in_distance_monotonic_per_head=monotonic,
            training=json.loads((REC/f'train_{arm}.json').read_text()))
    winner = min(ARMS,key=lambda a:(round(arms[a]['validation']['nll'],6),ARMS.index(a)))
    result = dict(status='completed',control=control,control_validation=reference,
                  arms=arms,winner=winner,
                  control_tokens=control['tokens'],arm_tokens=STEPS*32768,
                  endpoint_token_difference=STEPS*32768-control['tokens'],
                  exact_matched_tokens=False,
                  caveats=['X34 heads already have unequal initialized parameters.',
                    'Age-1 write addressing can synchronize occupancy despite different reads.',
                    'B64 vs historical B8 changes optimizer update count.',
                    'Endpoint differs by 4096 tokens; quality comparison is near-matched.',
                    'Widened base_lm controls are parameter receipts, not newly trained baselines.',
                    'Policy hierarchy changes budget, pop and decay together; no causal isolation.'])
    write(REC/'summary.json',result)
    return winner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('audit','seal-gate','train','all','summarize'))
    parser.add_argument('--arm',choices=ARMS,default='b')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command=='audit':
        print(json.dumps(audit()));return
    if args.command=='seal-gate':
        write(REC/'gate_sources.json',sources());gate();return
    if args.command=='summarize':
        print(summarize());return
    held = os.environ.get('GPU_LEASE_HELD')
    if held not in ('0','1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES') != held:
        raise RuntimeError('single whitelisted broker lease required')
    tok = Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    if args.command=='train':
        train(args.arm,tok);return
    audit()
    for arm in ARMS:
        model = train(arm,tok)
        del model
        torch.cuda.empty_cache()
    winner = summarize()
    model = LM(winner,tok.get_vocab_size()).cuda()
    saved = torch.load(ROOT/'runs'/winner/'latest.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(saved['model']);del saved
    winner_probe(model,winner,tok)


if __name__=='__main__':
    main()
