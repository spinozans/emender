"""Constructive CPU receipt for the registered detached read-match wire.

This is a lane-wiring witness, not an optimized implementation or training.
The certified oracle operations and thresholded, detached gate are unchanged.
"""
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from ..x30_tape_cell.naive_reference import TapeHeadConfig, init_state, tape_scan
from .audit import TASKS, digest


def match_receipt():
    n, d, heads = 8, 16, 4
    aliases = torch.eye(d)
    shared = torch.cat((torch.eye(n), torch.eye(n)), 1)
    rows = []
    for semantic, wired in ((False, True), (True, False), (True, True)):
        features = F.pad(F.linear(aliases, shared), (0, n)) if semantic else aliases
        cfg = TapeHeadConfig(key_dim=d, value_dim=d, n_heads=heads, budget=32)
        state = init_state(1, cfg, 'cpu')
        enabled, allocated, evidence = [], [], []
        for t in range(2*d):
            key = torch.zeros(1, 1, heads, d)
            value = torch.zeros_like(key)
            if t % 2 == 0:
                i = t//2
                key[:] = 8*features[i]
                value[..., i % n] = 1
            # Same scaled key/query score used by READ, before POP/WRITE.
            # Empty stores supply zero evidence, rather than a masked sentinel.
            scores = (state.keys * key[:, 0, :, None]).sum(-1)/math.sqrt(d)
            confidence = scores.masked_fill(~state.live, -float('inf')).amax(-1)
            confidence = torch.where(state.live.any(-1), confidence, 0.).detach()
            gate_logit = 2. - (confidence if wired else torch.zeros_like(confidence))
            beta = (torch.sigmoid(gate_logit).detach() > .5).float()
            if t % 2:
                beta.zero_()
            _, state, trace = tape_scan(key, key, value, beta[:, None],
                torch.zeros(1, 1, heads), torch.full((1, 1, heads), 20.), state, cfg)
            if t % 2 == 0:
                enabled.append(float(beta.mean()))
                allocated.append(float((trace.alloc_target >= 0).float().mean()))
                evidence.append(float(confidence.mean()))
        rows.append(dict(semantic_keys=semantic, match_wire=wired,
            novel_enabled=sum(enabled[:n])/n, repeat_enabled=sum(enabled[n:])/n,
            novel_allocation=sum(allocated[:n])/n,
            repeat_allocation=sum(allocated[n:])/n, match_evidence=evidence))
    assert rows[0]['repeat_enabled'] == rows[1]['repeat_enabled'] == 1.
    assert rows[2]['novel_enabled'] == rows[2]['novel_allocation'] == 1.
    assert rows[2]['repeat_enabled'] == rows[2]['repeat_allocation'] == 0.
    return dict(status='passed', rows=rows,
        construction='W_write h = 2; match_coef = -1; detached pre-read max scaled match; empty evidence=0; POP=0',
        deviation='X39-local sequential read-match-to-write-enable wiring; oracle operations untouched',
        limits='Hand-set constructive witness only. Error coefficient is zero in this witness. No learned selectivity, optimized lane parity, or LM quality established.')


def main():
    torch.set_num_threads(1)
    receipt = dict(match_gate=match_receipt(), training='not-run', gpu_parity='not-run',
        source_sha256={p: digest(TASKS/p) for p in (
            'x39_memory_usage/part4_match.py', 'x30_tape_cell/naive_reference.py')})
    path = Path(__file__).parent/'receipts/part4_match.json'
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
