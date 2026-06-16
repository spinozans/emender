"""GPU smoke test for the M1 state-aware MLP wiring on the FUSED E97 kernel.

NON-NEGOTIABLE #1: the recurrence runs through the fused Triton kernel (use_triton,
bf16). This test builds the grok E97 cell with the state-aware MLP (M1) enabled,
runs one fwd+bwd step under bf16 autocast, asserts the fused kernel engaged
(no eager fallback), and verifies gradients flow into the readout_summary module
(so the state-aware feature is actually wired into the loss, not inert).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F
from ndm.models.hybrid_ladder import HybridLadderLM

assert torch.cuda.is_available(), "smoke needs CUDA (fused kernel)"
dev = 'cuda'
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')

dim, depth, n_state, n_heads, vocab = 256, 4, 32, 8, 48


def build(state_summary_dim, mlp_hidden):
    return HybridLadderLM(
        vocab_size=vocab, dim=dim, depth=depth,
        layer_pattern=['E97'],
        layer_kwargs=[dict(state_activation='tanh', use_split_edit=True)],
        n_state=n_state, n_heads=n_heads, expansion=1.0,
        mlp_hidden=mlp_hidden, use_triton_e88=True,
        state_summary_dim=state_summary_dim,
    ).to(dev)


for tag, ssd, mh in [('baseline', 0, 1024), ('m1b', 128, 736), ('control', 0, 1024)]:
    m = build(ssd, mh)
    # fused-guard: every E97 mixer must be on the Triton kernel
    for layer in m.layers:
        assert getattr(layer, 'use_triton', False), f"{tag}: E97 layer not on Triton"
    assert all(m._is_e88_layer), f"{tag}: missing E97 layers"
    assert m.cast_recurrent_bf16, f"{tag}: bf16 cast off (fused inert)"
    npar = sum(p.numel() for p in m.parameters())
    x = torch.randint(0, vocab, (4, 128), device=dev)
    m.train()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = m(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), x[:, 1:].reshape(-1))
    loss.backward()
    # readout_summary must receive gradient when M1 is enabled
    if ssd > 0:
        g = m.layers[0].readout_summary.weight.grad
        assert g is not None and g.abs().sum().item() > 0, f"{tag}: readout_summary grad is None/zero (INERT)"
        ssd_msg = f"readout_summary.grad_norm={g.norm().item():.3e}"
    else:
        assert m.layers[0].readout_summary is None, f"{tag}: readout_summary unexpectedly built"
        ssd_msg = "no state summary"
    print(f"[fused-guard PASS] {tag:9s} params={npar:,} ssd={ssd} mlp_hidden={mh} "
          f"loss={loss.item():.4f} {ssd_msg}", flush=True)

print("SMOKE OK: fused E97 + M1 state-aware MLP trains, grads flow, no eager fallback.")
