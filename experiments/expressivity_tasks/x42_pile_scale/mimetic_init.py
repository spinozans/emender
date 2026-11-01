"""Standing donor initialization; copied from the receipted X42 transplant."""
import hashlib
from pathlib import Path
import torch
from .velocity_state import atomic_json
ROOT=Path(__file__).resolve().parent
DONOR=ROOT/'receipts/mimetic_geometry/donor_weights.pt'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

@torch.no_grad()
def transplant(model, checkpoint, receipt):
    state = checkpoint['state_dict']
    rows = []
    allowed = set()
    before = {n:tensor_sha(p) for n,p in model.named_parameters()}
    for i, block in enumerate(model.blocks):
        prefix = f'model.blocks.{i}.'
        qkv = state[prefix+'qkv.weight']; bias = state[prefix+'qkv.bias']
        d = qkv.shape[1]
        # Capacity geometry: shared orthogonal input coordinates chosen from
        # donor K/V, then rank-limited factorization of donor K content map.
        _, _, vh = torch.linalg.svd(torch.cat((qkv[d:2*d], qkv[2*d:]),0), full_matrices=False)
        basis = vh[:256]
        block.input.weight.copy_(basis); block.input.bias.zero_()
        kmap = qkv[d: d+64] @ basis.T
        u,s,v = torch.linalg.svd(kmap, full_matrices=False)
        rank = min(block.cell.policy.w_in_width,64)
        block.cell.W_in[0].weight.zero_(); block.cell.W_in[1].weight.zero_()
        block.cell.W_in[0].weight[:rank].copy_(s[:rank,None].sqrt()*v[:rank])
        block.cell.W_in[1].weight[:,:rank].copy_(u[:,:rank]*s[:rank].sqrt())
        # Evenly spaced complete 64-d donor heads; two per 128-d memory value.
        heads = torch.linspace(0,11,2*model.policy.N).round().long()
        indices = (heads[:,None]*64+torch.arange(64)).flatten()
        value = qkv[2*d:][indices]
        block.cell.W_v.weight.copy_(value@basis.T)
        block.cell.W_v.bias.copy_(bias[2*d:][indices])
        block.read_out.weight.copy_(state[prefix+'out.weight'][:,indices])
        block.read_out.bias.copy_(state[prefix+'out.bias'])
        donor_hidden = state[prefix+'ff.0.weight'].shape[0]
        recipient_hidden = block.ff[0].out_features
        neurons = torch.arange(recipient_hidden)%donor_hidden
        multiplicity = torch.bincount(neurons,minlength=donor_hidden).clamp_min(1)
        block.ff[0].weight.copy_(state[prefix+'ff.0.weight'][neurons])
        block.ff[0].bias.copy_(state[prefix+'ff.0.bias'][neurons])
        block.ff[2].weight.copy_(state[prefix+'ff.2.weight'][:,neurons]/multiplicity[neurons])
        block.ff[2].bias.copy_(state[prefix+'ff.2.bias'])
        names = ['input.weight','input.bias','cell.W_in.0.weight','cell.W_in.1.weight',
                 'cell.W_v.weight','cell.W_v.bias','read_out.weight','read_out.bias',
                 'ff.0.weight','ff.0.bias','ff.2.weight','ff.2.bias']
        allowed.update(f'blocks.{i}.'+n for n in names)
        rows.append(dict(layer=i,donor_width=d,recipient_width=block.input.in_features,
            donor_heads=12,donor_head_indices=heads.tolist(),memory_heads=model.policy.N,
            input_rank=256,content_rank=rank,donor_mlp_hidden=donor_hidden,
            recipient_mlp_hidden=recipient_hidden,
            value_relative_projection_error=float(torch.linalg.vector_norm(value-(value@basis.T)@basis)/torch.linalg.vector_norm(value)),
            content_relative_projection_error=float(torch.linalg.vector_norm(qkv[d:d+64]-kmap@basis)/torch.linalg.vector_norm(qkv[d:d+64]))))
    after = {n:tensor_sha(p) for n,p in model.named_parameters()}
    changed = [n for n in before if before[n]!=after[n]]
    assert set(changed)<=allowed
    assert all(before[n]==after[n] for n,p in model.named_parameters() if not p.requires_grad)
    atomic_json(receipt,dict(donor=str(DONOR),donor_sha256=sha(DONOR),
        copied=sorted(allowed),changed=changed,layers=rows,frozen_unchanged=True,
        untouched='queries, semantic embeddings, key projections, management, embedding/head, layer norms',
        scope='partial read-path transplant; geometry projection is lossy; not function-equivalent attention'))

