#!/usr/bin/env python3
"""fix-long-horizon: held-out BPB for one checkpoint (corrected-recipe runs).

Identical metric to the predecessor diloco-loss-parity-longhorizon ckpt_bpb.py
(same agent-1433 preflight held-out tensor, bytes/token 3.938, 65536 tokens,
strict=True reload of the emender-1.286B consensus/DDP checkpoint) so the
corrected-recipe curve is directly comparable to the broken-recipe baseline.

REAL checkpoint, REAL held-out data. One emit line per call:
  BPB_RESULT <tag> <step> <bpb> <ce> <tokens>
"""
import os, sys, math, argparse
import torch

# Use THIS worktree's ndm (agent-1442) — same model code, keeps the run hermetic.
REPO = '/home/erikg/ndm/.wg-worktrees/agent-1442'
sys.path.insert(0, REPO)
from ndm.models import LadderLM

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--heldout', required=True)
p.add_argument('--tag', default='run')
args = p.parse_args()

ck = torch.load(args.ckpt, map_location='cpu')
step = int(ck.get('step', 0))

device = torch.device('cuda')
model = LadderLM(
    vocab_size=50281, dim=1792, depth=11, level='E97',
    expansion=1.0, n_state=32, n_heads=216,
    use_gate=True, gate_activation='silu', e88_raw_write=False,
    use_triton=True, mlp_ratio=2.262336203876648, mlp_multiple=64,
).to(device).bfloat16()

sd = ck['model_state_dict']
missing, unexpected = model.load_state_dict(sd, strict=True)
assert not missing and not unexpected, f'state_dict mismatch missing={list(missing)} unexpected={list(unexpected)}'

ho = torch.load(args.heldout, map_location='cpu')
chunks = ho['chunks']; bpt = float(ho['bytes_per_token'])
model.eval()
tot_nll = 0.0; tot = 0
with torch.no_grad():
    for i in range(0, chunks.shape[0], 4):
        b = chunks[i:i+4].to(device)
        loss = model(b, return_loss=True)
        loss = loss[0] if isinstance(loss, tuple) else loss
        n = b.shape[0]*(b.shape[1]-1)
        tot_nll += float(loss.item())*n; tot += n
ce = tot_nll/max(tot, 1)
bpb = (ce/math.log(2.0))/bpt
assert math.isfinite(ce), 'non-finite reload loss'
print(f'BPB_RESULT {args.tag} {step} {bpb:.4f} {ce:.4f} {tot}', flush=True)
