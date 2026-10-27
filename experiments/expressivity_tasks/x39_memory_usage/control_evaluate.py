"""Usage telemetry and preregistered probes for the B64 family control."""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from experiments.expressivity_tasks.x30_tape_cell.routing_trace import CASE_NAMES
from experiments.expressivity_tasks.x34_lm_pilot import data
from experiments.expressivity_tasks.x34_lm_pilot.cli import validate, write
from experiments.expressivity_tasks.x34_lm_pilot.probes import (
    aggregate_depth, context_probe, depth_probe, story_probe)
from experiments.expressivity_tasks.x36_tape_perf.model import LM

ROOT = Path(__file__).resolve().parent


def profile(state, trace, pressure):
    allocated = (trace.alloc_target >= 0).long()
    live = (allocated-(trace.freed != 0).sum(-1)).cumsum(1)
    peak = live+(trace.freed == 2).sum(-1)
    assert torch.equal(live[:, -1], state.native.live.sum(-1))
    cap_codes = [code for code, name in CASE_NAMES.items() if name == 'R-CAP']
    assert len(cap_codes) == 1
    cap = trace.case == cap_codes[0]
    requests = allocated.bool() | cap
    w64 = cap.float().unfold(1,64,1).mean(-1) if live.shape[1] >= 64 else None
    return dict(mean_live_per_head=live.float().mean((0,1)).tolist(),
        peak_live_per_head=peak.amax((0,1)).tolist(),
        final_live_per_head=live[:, -1].amax(0).tolist(),
        occupancy_profile=live.float().mean(0).tolist(),
        allocation_rate_per_token_head=allocated.float().mean((0,1)).tolist(),
        append_request_rate_per_token_head=requests.float().mean((0,1)).tolist(),
        write_enabled_fraction_per_head=trace.w_on.float().mean((0,1)).tolist(),
        pressure_fraction_per_head=(pressure != 0).float().mean((0,1)).tolist(),
        pressure_mean_per_head=pressure.mean((0,1)).tolist(),
        pressure_max_per_head=pressure.amax((0,1)).tolist(),
        rcap_fraction_per_head=cap.float().mean((0,1)).tolist(),
        w64_saturation_fraction_per_head=None if w64 is None else
            (w64 >= .9).float().mean((0,1)).tolist())


def main():
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):
        raise RuntimeError('X39 broker lease required')
    torch.set_num_threads(1)
    saved = torch.load(ROOT/'runs/control_b64/latest.pt',map_location='cpu',weights_only=False)
    assert saved['step'] == 3052
    for name, expected in saved['fingerprint']['sources'].items():
        path = (ROOT/'runs/control_b64/registered_protocol.md'
                if name == 'x39_memory_usage/RECEIPTS.md' else ROOT.parent/name)
        assert data.sha(path) == expected, name
    for name, filename in (('tokenizer','tokenizer.json'),('valid','valid.bin')):
        assert data.sha(data.RUNS/filename) == saved['fingerprint'][name], name
    tok = Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    model = LM('tape_lm',tok.get_vocab_size()).cuda().eval()
    model.load_state_dict(saved['model'])
    del saved
    current = {}

    def capture(cell, args, output):
        _, state, trace, pressure = output
        current.clear()
        current.update(profile(state,trace,pressure))
        h, ids = args
        current['sigmoid_gate_strength_per_head'] = cell.W_write(h).sigmoid().mean((0,1)).tolist()

    def attach(module, args, output):
        output[1][0].update(current)

    model.blocks[2].cell.register_forward_hook(capture)
    model.register_forward_hook(attach)
    valid = np.memmap(data.RUNS/'valid.bin',mode='r',dtype=np.uint16)
    with torch.no_grad():
        validation = validate(model,'cuda')
        contexts = context_probe(model,valid,'cuda')
        stories = story_probe(model,tok,data.RUNS/'valid.txt','cuda',stories=16)
        depth = depth_probe(model,tok,'cuda')
    slopes = []
    for head in range(4):
        peaks = [max(m[0]['peak_live_per_head'][head] for m in row['telemetry'])
                 for row in contexts]
        slopes.append(None if min(peaks) <= 0 else
                      float(np.polyfit(np.log([512,1024,2048,4096]),np.log(peaks),1)[0]))
    result = dict(status='completed',tokens=100007936,step=3052,
        physical_gpu=os.environ['GPU_LEASE_HELD'], validation=validation,
        contexts=contexts, occupancy_log_log_slopes_per_head=slopes,
        stories=stories,depth=depth,depth_bins=aggregate_depth(depth),
        source_sha256=data.sha(Path(__file__)),
        checkpoint_sha256=data.sha(ROOT/'runs/control_b64/latest.pt'))
    write(ROOT/'receipts/part6_control_evaluation.json',result)
    print(json.dumps(dict(perplexity=validation['perplexity'],slopes=slopes)),flush=True)


if __name__ == '__main__':
    main()
