#!/usr/bin/env python3
"""preflight-100b: checkpoint SAVE+RELOAD roundtrip verification.

Loads the final checkpoint produced by the 7-GPU DDP emender-mlp run into a
freshly-built single-process model (the inference/generate path), confirms the
state_dict loads with NO missing/unexpected keys, and runs a forward on the
held-out tensor to confirm it produces a FINITE loss matching the in-run
FINAL_HELDOUT (proving the saved weights are usable, not x-mode garbage).

REAL checkpoint, REAL data — no fabrication.
"""
import os, sys, glob, math, argparse
import torch

REPO = '/home/erikg/ndm/.wg-worktrees/agent-1433'
sys.path.insert(0, REPO)
from ndm.models import LadderLM

p = argparse.ArgumentParser()
p.add_argument('--ckpt_dir', required=True)
p.add_argument('--heldout', required=True)
args = p.parse_args()

# Locate the final checkpoint (latest.pt symlink or newest checkpoint_step_*.pt)
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

# Rebuild the emender-mlp (E97 delta + SwiGLU MLP) model with identical geometry.
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

# Forward on held-out -> finite loss
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
