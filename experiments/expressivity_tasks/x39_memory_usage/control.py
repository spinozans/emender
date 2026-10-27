"""Exact-exposure B64 control; invoke only through X39 run_leased.sh."""
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from ..x34_lm_pilot import data
from ..x34_lm_pilot.cli import replay_present, save_checkpoint, validate, write
from ..x34_lm_pilot.model import parameter_receipt
from ..x36_tape_perf.model import LM
from ..x36_tape_perf.steady import FiniteGradients

ROOT = Path(__file__).resolve().parent
REC = ROOT / 'receipts'
RUN = ROOT / 'runs' / 'control_b64'
STEPS = 3052


def fingerprint():
    paths = list(ROOT.glob('*.py')) + [ROOT/'RECEIPTS.md', ROOT/'run_leased.sh']
    for name in ('x30_tape_cell', 'x31_tape_cell_fused', 'x34_lm_pilot', 'x36_tape_perf'):
        paths.extend((ROOT.parent/name).glob('*.py'))
    return dict(seed=3401, batch=64, context=512, steps=STEPS,
        sources={str(p.relative_to(ROOT.parent)): data.sha(p) for p in sorted(paths)},
        tokenizer=data.sha(data.RUNS/'tokenizer.json'),
        train=data.sha(data.RUNS/'train.bin'), valid=data.sha(data.RUNS/'valid.bin'))


def train():
    if os.environ.get('GPU_LEASE_HELD') not in ('0', '1', '5', '6'):
        raise RuntimeError('X39 broker lease required')
    # A fresh passing production-shape forward/backward gate is mandatory.
    import xml.etree.ElementTree as ET
    cases = ET.parse(REC/'part6_control_parity.xml').getroot().findall('.//testcase')
    if len(cases) != 1 or any(list(c) for c in cases):
        raise RuntimeError('B64 production-shape oracle gate did not pass')
    torch.set_num_threads(1)
    torch.manual_seed(3401); np.random.seed(3401); random.seed(3401)
    tok = Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    write(REC/'part6_control_parameters.json', parameter_receipt(tok.get_vocab_size()))
    fp = fingerprint()
    model = LM('tape_lm', tok.get_vocab_size()).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), weight_decay=.1)
    RUN.mkdir(parents=True, exist_ok=True)
    checkpoint = RUN/'latest.pt'
    step, elapsed = 0, 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if saved['fingerprint'] != fp:
            raise RuntimeError('control resume fingerprint mismatch')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        step, elapsed = saved['step'], saved['elapsed_seconds']
        torch.set_rng_state(saved['rng'].cpu()); torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        del saved
    write(RUN/'registration.json', fp)
    finite = FiniteGradients('cuda')
    start = time.monotonic()
    status = 'failed'
    try:
        with (RUN/'curve.jsonl').open('a', buffering=1) as log:
            while step < STEPS:
                x, y = (a.pin_memory().cuda(non_blocking=True)
                        for a in data.batch('train', step, batch_size=64))
                opt.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.flatten(0,1), y.flatten())
                replay_present(loss)
                loss.backward()
                norm = finite.clip(model.parameters())
                opt.step()
                step += 1
                # This synchronized curve prioritizes auditable finite boundaries.
                finite.check()
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError('nonfinite control loss')
                log.write(json.dumps(dict(step=step, tokens=step*32768, loss=value,
                    grad_norm=float(norm), replay_node='ReplayBackward'))+'\n')
                del logits, loss
                if step % 64 == 0 or step in (611, STEPS):
                    if step % 256 == 0 or step in (611, STEPS):
                        write(REC/f'control_b64_{step}.json', dict(step=step,
                            tokens=step*32768, validation=validate(model, 'cuda')))
                    save_checkpoint(checkpoint, dict(fingerprint=fp, step=step,
                        model=model.state_dict(), optimizer=opt.state_dict(),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        elapsed_seconds=elapsed+time.monotonic()-start))
                    print(json.dumps(dict(step=step,tokens=step*32768,
                        elapsed_seconds=elapsed+time.monotonic()-start)),flush=True)
        status = 'completed'
    except BaseException as exc:
        write(REC/'part6_control_failure.json', dict(step=step,error=repr(exc)))
        raise
    finally:
        write(REC/'part6_control_training.json', dict(status=status, step=step,
            tokens=step*32768, fingerprint=fp, physical_gpu=os.environ['GPU_LEASE_HELD'],
            elapsed_seconds=elapsed+time.monotonic()-start))


if __name__ == '__main__':
    train()
