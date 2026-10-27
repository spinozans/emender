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
MILESTONES = (0,64,512,611,1536,STEPS)


def sources():
    paths = list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    paths += [ROOT/'RECEIPTS.md',ROOT/'run_leased.sh']
    paths += [REC/'resume_pinned.py']
    for folder in ('x30_tape_cell','x31_tape_cell_fused','x34_lm_pilot','x36_tape_perf','x37_head_differentiation'):
        paths += list((ROOT.parent/folder).glob('*.py'))
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
    for name,minimum in (('oracle.xml',179),('mechanism.xml',10)):
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
    model.measure_slots = True
    stories = story_probe(model,tok,data.RUNS/'valid.txt','cuda',stories=16)
    model.measure_slots = False
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


def train(arm,tok, target=STEPS):
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
            while step<target:
                tic = time.perf_counter()
                x,y = (a.pin_memory().cuda(non_blocking=True)
                       for a in data.batch('train',step,batch_size=64))
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                optimizer.zero_grad(set_to_none=True)
                logits,_ = model(x)
                loss = model.objective(logits,y)
                replay_present(loss)
                loss.backward()
                norm = finite.clip(model.parameters())
                optimizer.step();end.record();step += 1
                pending.append((step,loss.detach(),norm.detach(),start,end,time.perf_counter()-tic))
                del logits,loss
                if step%64 == 0 or step == target:
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


def select_sweep():
    rows = {a:json.loads((REC/f'eval_{a}_611.json').read_text()) for a in ARMS}
    baseline = rows['0']['validation']['perplexity']
    eligible = [a for a in ARMS if rows[a]['validation']['perplexity'] <= 1.02*baseline]
    def correlation(a):
        return statistics.mean(s['telemetry'][0]['read_cosine_mean'] for s in rows[a]['stories'])
    selected = min(eligible,key=lambda a:(correlation(a),float(a)))
    write(REC/'sweep.json',dict(selected=selected,rows=rows,rule='min correlation within 2% lambda0 perplexity'))
    return selected


def summarize():
    selected = json.loads((REC/'sweep.json').read_text())['selected']
    final = json.loads((REC/f'eval_{selected}_{STEPS}.json').read_text())
    reference = json.loads((data.ROOT/'receipts/pilot_tape_lm.json').read_text())['curve'][-1]['validation']
    ratio = final['validation']['perplexity']/reference['perplexity']
    result = dict(selected=selected,final=final,perplexity_ratio_to_a=ratio,
                  quality_within_2pct=abs(ratio-1)<=.02,quality_over_5pct_worse=ratio>1.05,
                  exact_matched_tokens=False,slot_usage_js_status='see per-story shadow replay telemetry')
    history = [json.loads((REC/f'eval_{selected}_{i}.json').read_text()) for i in MILESTONES]
    def telemetry_mean(row,key):
        values = [s['telemetry'][0][key] for s in row['stories']]
        if key == 'slot_usage_js':
            values = [v for d in values for v in d.values() if v is not None]
        return statistics.mean(values)
    correlation = [telemetry_mean(row,'read_cosine_mean') for row in history]
    js = [telemetry_mean(row,'slot_usage_js') for row in history]
    monotonic = lambda values:all(b>=a for a,b in zip(values,values[1:]))
    pair_keys = list(history[0]['W_in']['pairwise_distance'])
    result.update(read_cosine_history=correlation,read_cosine_monotonic_decrease=monotonic([-v for v in correlation]),
        final_read_cosine_below_point3=correlation[-1]<.3,
        slot_js_history=js,slot_js_monotonic_increase=monotonic(js),
        W_in_pairwise_monotonic={k:monotonic([r['W_in']['pairwise_distance'][k] for r in history]) for k in pair_keys},
        W_in_from_init_monotonic=[monotonic([r['W_in']['distance_from_init'][h] for r in history]) for h in range(4)],
        W_in_pairs_exceed_init={k:history[-1]['W_in']['pairwise_distance'][k]>history[0]['W_in']['pairwise_distance'][k] for k in pair_keys},
        expected_occupancy_mirror=final['primary']['identical_stories']==16,
        honest_null_decorrelation_bought_too_dearly=correlation[-1]<.3 and ratio>1.02)
    write(REC/'summary.json',result)
    (ROOT/'REPORT.md').write_text(
        '# X38 measured result\n\n'
        f'Selected lambda: {selected}. Final tokens: {STEPS*32768}.\n'
        f'Read cosine: {correlation[0]:.6f} -> {correlation[-1]:.6f}; '
        f'monotonic decrease: {result["read_cosine_monotonic_decrease"]}.\n'
        f'Slot JS bits: {js[0]:.6f} -> {js[-1]:.6f}.\n'
        f'Perplexity ratio to historical a: {ratio:.6f}; '
        f'within2%: {result["quality_within_2pct"]}; over5% worse: {result["quality_over_5pct_worse"]}.\n'
        f'Identical occupancy stories: {final["primary"]["identical_stories"]}/16 (expected).\n\n'
        'See receipts/summary.json for all monotonicity tests and failures. '
        'Control is B8 and4096 tokens shorter: not an exact matched-token comparison. '
        'Slot identity compares physical allocation indices. No causal semantic specialization claim.\n')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('audit','seal-gate','train','all','summarize'))
    parser.add_argument('--arm',choices=ARMS,default='0.01')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command=='audit':
        print(json.dumps(audit()));return
    if args.command=='seal-gate':
        write(REC/'gate_sources.json',sources());gate()
        import tarfile
        with tarfile.open(REC/'gated_sources.tar.gz','w:gz') as archive:
            for name in sources():
                archive.add(ROOT.parent/name,arcname=name)
        return
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
        model = train(arm,tok,611)
        del model
        torch.cuda.empty_cache()
    winner = select_sweep()
    model = train(winner,tok)
    del model
    summarize()
    model = LM(winner,tok.get_vocab_size()).cuda()
    saved = torch.load(ROOT/'runs'/winner/'latest.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(saved['model']);del saved
    winner_probe(model,winner,tok)


if __name__=='__main__':
    main()
