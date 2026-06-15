#!/usr/bin/env python3
"""implement-diloco-periodic: checkpoint SAVE+RELOAD roundtrip for the DiLoCo
CONSENSUS checkpoint. After the final cross-rank merge, rank 0 saves the
schedule-free EVAL (averaged x) weights. This verifies that saved consensus model
loads strict=True into a fresh single-process model and produces a FINITE held-out
loss -> proves the y-mode merge produced usable weights, not x-mode garbage, and
that the DiLoCo checkpoint is interchangeable with the normal inference path.

REAL checkpoint (from the 7-GPU DiLoCo run), REAL held-out data.
"""
import os, sys, glob, math, argparse
import torch

REPO = '/home/erikg/ndm/.wg-worktrees/agent-1436'
sys.path.insert(0, REPO)
from ndm.models import LadderLM

p = argparse.ArgumentParser()
p.add_argument('--ckpt_dir')
p.add_argument('--ckpt', help='explicit checkpoint .pt (overrides --ckpt_dir)')
p.add_argument('--heldout', required=True)
args = p.parse_args()

if args.ckpt:
    cand = args.ckpt
else:
    cand = os.path.join(args.ckpt_dir, 'latest.pt')
    if not os.path.exists(cand):
        runs = sorted(glob.glob(os.path.join(args.ckpt_dir, '**', 'checkpoint_step_*.pt'), recursive=True))
        assert runs, f'no checkpoint under {args.ckpt_dir}'
        cand = runs[-1]
    else:
        cand = os.path.realpath(cand)
print(f'CKPT: {cand}')
ck = torch.load(cand, map_location='cpu')
print(f'CKPT keys: step={ck.get("step")} loss={ck.get("loss"):.4f} '
      f'n_state_tensors={len(ck["model_state_dict"])}')

device = torch.device('cuda')
model = LadderLM(
    vocab_size=50281, dim=1792, depth=11, level='E97',
    expansion=1.0, n_state=32, n_heads=216,
    use_gate=True, gate_activation='silu', e88_raw_write=False,
    use_triton=True, mlp_ratio=2.262336203876648, mlp_multiple=64,
).to(device).bfloat16()

sd = ck['model_state_dict']
missing, unexpected = model.load_state_dict(sd, strict=True)
print(f'LOAD_OK missing={list(missing)} unexpected={list(unexpected)}')
assert not missing and not unexpected, 'state_dict mismatch on reload'

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
ce = tot_nll/max(tot,1)
bpb = (ce/math.log(2.0))/bpt
assert math.isfinite(ce), 'non-finite reload loss'
print(f'RELOAD_HELDOUT_CE: {ce:.4f}')
print(f'RELOAD_HELDOUT_BPB: {bpb:.4f}')
print(f'RELOAD_HELDOUT_TOKENS: {tot}')
print('ROUNDTRIP_OK')
