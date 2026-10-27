"""Sequential X39 usage runs; all GPU execution requires run_leased.sh.

Arm 4 is deliberately not substituted: the archived witness does not specify
an optimized recurrent match-to-gate implementation.
"""
from dataclasses import replace, asdict
import json
import math
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from ..x34_lm_pilot import data
from ..x34_lm_pilot.cli import replay_present, save_checkpoint, validate, write
from ..x34_lm_pilot.model import parameter_receipt
from ..x34_lm_pilot.probes import context_probe, story_probe, depth_probe, aggregate_depth
from ..x36_tape_perf.model import LM
from ..x36_tape_perf.steady import FiniteGradients
from .control import fingerprint
from .control_evaluate import profile

ROOT = Path(__file__).resolve().parent
REC = ROOT / 'receipts'


def configure(model, arm):
    cell = model.blocks[2].cell
    if arm == 'decay':
        cell.cfg = replace(cell.cfg, decay_rate=float(np.float32(math.exp(math.log(1e-3)/128))))
    elif arm == 'poverty':
        cell.cfg = replace(cell.cfg, budget=128)
    elif arm != 'gate':
        raise ValueError(arm)
    return cell


def parity(cfg):
    # Reuse the full forward/state/trace/gradient bitwise comparison at the
    # actual B64/H4/K64/V128/T512 shape, with the arm's exact configuration.
    from ..x36_tape_perf.tests import test_exact
    with patch.object(test_exact, 'TapeHeadConfig', lambda **kwargs: cfg):
        test_exact.test_fresh_exact(64, cfg.slots, 512, 64, 128, False)


def evaluate(model, path, checkpoint, tokens, budget=None):
    cell = model.blocks[2].cell
    original = cell.cfg
    if budget is not None:
        cell.cfg = replace(original, budget=budget)
    current = {}
    def capture(module, args, output):
        _, state, trace, pressure = output
        current.clear()
        current.update(profile(state, trace, pressure))
        current['sigmoid_gate_strength_per_head'] = module.W_write(args[0]).sigmoid().mean((0,1)).tolist()
    def attach(module, args, output):
        output[1][0].update(current)
    hooks = [cell.register_forward_hook(capture), model.register_forward_hook(attach)]
    model.eval()
    try:
        valid = np.memmap(data.RUNS/'valid.bin', mode='r', dtype=np.uint16)
        tok = Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
        with torch.no_grad():
            quality = validate(model, 'cuda')
            contexts = context_probe(model, valid, 'cuda')
            stories = story_probe(model, tok, data.RUNS/'valid.txt', 'cuda', stories=16)
            depth = depth_probe(model, tok, 'cuda')
        slopes = []
        for h in range(4):
            peaks = [max(m[0]['peak_live_per_head'][h] for m in row['telemetry']) for row in contexts]
            slopes.append(None if min(peaks) <= 0 else float(np.polyfit(np.log([512,1024,2048,4096]), np.log(peaks), 1)[0]))
        reference = json.loads((REC/'control_b64_3052.json').read_text())['validation']['perplexity']
        ratio = quality['perplexity']/reference
        result = dict(status='completed', tokens=tokens, budget=cell.cfg.budget,
            validation=quality, perplexity_ratio=ratio, within_2pct=ratio <= 1.02,
            within_5pct=ratio <= 1.05, occupancy_log_log_slopes_per_head=slopes,
            contexts=contexts, stories=stories, depth=depth, depth_bins=aggregate_depth(depth),
            checkpoint_sha256=data.sha(checkpoint), physical_gpu=os.environ['GPU_LEASE_HELD'])
        write(path, result)
        print(json.dumps(dict(evaluation=str(path), ppl=quality['perplexity'], ratio=ratio, slopes=slopes)), flush=True)
        return result
    finally:
        for hook in hooks:
            hook.remove()
        cell.cfg = original


def train(arm, cost=0., pilot=False):
    tag = arm if arm != 'gate' else f'gate_{cost:g}' + ('_pilot' if pilot else '_full')
    run = ROOT/'runs'/tag
    run.mkdir(parents=True, exist_ok=True)
    steps = 611 if pilot else 3052
    torch.manual_seed(3401); np.random.seed(3401); random.seed(3401)
    tok = Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    # Match control.train's RNG-consuming parameter construction order exactly.
    counts = parameter_receipt(tok.get_vocab_size())
    model = LM('tape_lm', tok.get_vocab_size()).cuda().train()
    initial_rng = torch.get_rng_state()
    initial_cuda_rng = torch.cuda.get_rng_state()
    cell = configure(model, arm)
    fp = fingerprint()
    fp.update(arm=arm, cost=cost, steps=steps, cfg=asdict(cell.cfg))
    write(run/'registration.json', fp)
    try:
        parity(cell.cfg)
    except BaseException as exc:
        write(REC/f'part7_{tag}_parity.json', dict(status='failed', error=repr(exc), fingerprint=fp))
        raise
    write(REC/f'part7_{tag}_parity.json', dict(status='passed', configuration=asdict(cell.cfg),
        batch=64, context=512, comparison='X31 forward, state, trace and gradients bitwise', fingerprint=fp))
    counts['actual_arm'] = dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert counts['actual_arm'] == counts['tape_lm']
    write(REC/f'part7_{tag}_parameters.json', counts)
    # Parity must not advance the control-matched post-initialization RNG.
    torch.set_rng_state(initial_rng); torch.cuda.set_rng_state(initial_cuda_rng)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), weight_decay=.1)
    checkpoint = run/'latest.pt'
    step, elapsed = 0, 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if saved['fingerprint'] != fp:
            raise RuntimeError(f'{tag} resume fingerprint mismatch')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        step, elapsed = saved['step'], saved['elapsed_seconds']
        torch.set_rng_state(saved['rng'].cpu()); torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        del saved
    gate = []
    def capture(module, args, output):
        gate.append(output.sigmoid().mean())
    hook = cell.W_write.register_forward_hook(capture) if cost else None
    finite = FiniteGradients('cuda')
    start = time.monotonic()
    status = 'failed'
    try:
        with (run/'curve.jsonl').open('a', buffering=1) as log:
            while step < steps:
                gate.clear()
                x,y = (a.pin_memory().cuda(non_blocking=True) for a in data.batch('train', step, batch_size=64))
                opt.zero_grad(set_to_none=True)
                logits,_ = model(x)
                ce = F.cross_entropy(logits.flatten(0,1), y.flatten())
                penalty = gate[0] if cost else ce.new_zeros(())
                loss = ce + cost*penalty
                replay_present(loss)
                loss.backward()
                norm = finite.clip(model.parameters())
                opt.step(); step += 1
                finite.check()
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError('nonfinite usage loss')
                log.write(json.dumps(dict(step=step, tokens=step*32768, loss=value,
                    ce=float(ce.detach()), gate_mean=float(penalty.detach()), grad_norm=float(norm),
                    replay_node='ReplayBackward'))+'\n')
                gate.clear()
                del logits, loss, ce, penalty
                if step % 64 == 0 or step in (611, steps):
                    if step % 256 == 0 or step in (611, steps):
                        write(REC/f'part7_{tag}_{step}.json', dict(step=step, tokens=step*32768, validation=validate(model,'cuda')))
                        gate.clear()
                    save_checkpoint(checkpoint, dict(fingerprint=fp, step=step, model=model.state_dict(),
                        optimizer=opt.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        elapsed_seconds=elapsed+time.monotonic()-start))
                    print(json.dumps(dict(arm=tag, step=step, tokens=step*32768,
                        elapsed_seconds=elapsed+time.monotonic()-start)), flush=True)
        status = 'completed'
    except BaseException as exc:
        write(REC/f'part7_{tag}_failure.json', dict(step=step,error=repr(exc)))
        raise
    finally:
        if hook:
            hook.remove()
        gate.clear()
        write(REC/f'part7_{tag}_training.json', dict(status=status, step=step, tokens=step*32768,
            fingerprint=fp, physical_gpu=os.environ['GPU_LEASE_HELD'], elapsed_seconds=elapsed+time.monotonic()-start))
    if not pilot:
        evaluate(model, REC/f'part7_{tag}_evaluation.json', checkpoint, step*32768)
        if arm == 'poverty':
            evaluate(model, REC/'part7_poverty_rich_evaluation.json', checkpoint, step*32768, budget=2944)
    return json.loads((REC/f'part7_{tag}_{steps}.json').read_text())['validation']['perplexity']


def main():
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):
        raise RuntimeError('X39 broker lease required')
    torch.set_num_threads(1)
    train('decay')
    rows = [dict(cost=cost, perplexity=train('gate', cost, pilot=True)) for cost in (.001,.01,.1)]
    reference = json.loads((REC/'control_b64_611.json').read_text())['validation']['perplexity']
    qualified = [row['cost'] for row in rows if row['perplexity'] <= reference*1.02]
    selected = max(qualified) if qualified else None
    write(REC/'part7_lambda_selection.json', dict(rows=rows, reference=reference,
        matched_tokens=20021248, threshold=reference*1.02, selected=selected,
        disposition='selected' if qualified else 'honest-null: no qualifying lambda'))
    if selected is not None:
        train('gate', selected)
    train('poverty')
    write(REC/'part7_semantic_blocker.json', dict(status='not-run',
        reason='No optimized match-to-write gate exists in staged X39 code. The CPU witness uses 1/sqrt(K), certified READ 2/sqrt(K); recurrent wiring and error-term placement unresolved in REPORT.md.',
        supervisor_tool='unavailable after tool discovery; no alternate orchestration service started'))


if __name__ == '__main__':
    main()
