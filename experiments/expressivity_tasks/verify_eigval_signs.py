#!/usr/bin/env python3
"""Verify the eigenvalue-sign edits do EXACTLY what ARM A / ARM B intend.

Gold-standard functional probe (no re-derived formula): we extract the ACTUAL
per-token transition operator A_t the running code computes, by finite
differences on the real recurrence.

  S_1[i,v] = sum_j A[i,j] S0[j,v] + B[i,v]    (one step, linear in S0)

so with the layer in eval+fp32 (which routes through the serial PyTorch
fallback where the ARM-B clamp lives), we read off A_t column-by-column:
  B       = S_1(S0 = 0)
  A[:, j] = S_1(S0 with row j = 1) - B     (constant across v)
then eig(A_t) and report the along-key eigenvalue distribution.

ARM B prediction: baseline e88-linear has a NEGATIVE along-key eigenvalue
(decay-1 in (-1,0)); the clamp makes the most-negative eigenvalue ~0 (>=0).

ARM A is the fla GatedDeltaNet allow_neg_eigval flag (beta*=2): we confirm the
along-key eigenvalue g*(1-beta) goes negative when the flag is on, using the
exact fla parameterization on real S5 inputs.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ndm.models.e88_fla_hybrid import E88FLAHybrid
from experiments.expressivity_tasks.tasks import ALL_TASKS

try:
    from fla.layers.gated_deltanet import GatedDeltaNet as FLAGatedDeltaNet
except Exception:
    FLAGatedDeltaNet = None


def extract_At_e88(layer, x):
    """Functional finite-difference extraction of A_t for an E88 layer.

    x: [1, T, dim] fp32 on cuda. Returns A_full [T, H, n, n] (numpy)."""
    layer.eval()
    B, T, D = x.shape
    H, n, vd = layer.n_heads, layer.n_state, layer.head_v_dim
    with torch.no_grad():
        # B_t (the affine offset) = S_1 when S0 = 0
        S0 = [torch.zeros(B, n, vd, device=x.device, dtype=x.dtype) for _ in range(H)]
        _, S_zero = layer(x, hidden=S0)
        Bt = torch.stack(S_zero, dim=1)  # [B,H,n,vd]; but this is the FINAL state after T steps
        # For a clean single-step A_t we use T=1 so S_final = A_0 S0 + B_0.
        A = np.zeros((H, n, n), dtype=np.float64)
        for j in range(n):
            S0j = [torch.zeros(B, n, vd, device=x.device, dtype=x.dtype) for _ in range(H)]
            for h in range(H):
                S0j[h][:, j, :] = 1.0
            _, S_j = layer(x, hidden=S0j)
            Sj = torch.stack(S_j, dim=1)  # [B,H,n,vd]
            col = (Sj - Bt)[:, :, :, 0]    # [B,H,n] = A[:, j] (constant across v)
            A[:, :, j] = col[0].double().cpu().numpy()  # B=1
    return A  # [H, n, n]


def alongkey_eig_e88(layer, x_seq):
    """For each token in a real sequence, build A_t (T=1 single-step probe at
    that token's projections) and take the most-negative real eigenvalue as the
    along-key eigenvalue. Returns array over (T, H)."""
    eigs = []
    T = x_seq.shape[1]
    for t in range(T):
        xt = x_seq[:, t:t+1]  # [1,1,dim]
        A = extract_At_e88(layer, xt)  # [H,n,n]
        for h in range(A.shape[0]):
            ev = np.linalg.eigvals(A[h]).real
            eigs.append(ev.min())  # along-key is the distinct (smallest) eigenvalue
    return np.array(eigs)


def gdn_alongkey(layer, x):
    """Exact fla along-key eigenvalue g*(1-beta) on real inputs (after conv+proj
    the fla layer applies). We replicate fla's beta and g exactly."""
    mod = layer.gdn
    x = x.float()
    a = F.linear(x, mod.a_proj.weight.float())
    g = (-mod.A_log.float().exp() * F.softplus(a + mod.dt_bias.float())).exp()
    beta = torch.sigmoid(F.linear(x, mod.b_proj.weight.float()))
    if getattr(mod, 'allow_neg_eigval', False):
        beta = beta * 2.0
    alongk = g * (1.0 - beta)
    return alongk.reshape(-1).detach().cpu().numpy()


def summ(name, arr):
    return {
        'name': name, 'n': int(arr.size),
        'min': float(arr.min()), 'max': float(arr.max()), 'mean': float(arr.mean()),
        'p5': float(np.percentile(arr, 5)), 'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'frac_negative': float((arr < 0).mean()),
        'frac_lt_-0.1': float((arr < -0.1).mean()),
    }


def main():
    device = 'cuda'
    torch.manual_seed(0)
    task = ALL_TASKS['s5_permutation']()
    rng = np.random.default_rng(7)
    inp, _, _ = task.generate_batch(1, 24, rng)
    x_tok = torch.from_numpy(inp).to(device)  # token ids [1,24]

    print("=" * 70)
    print("ARM B  (E88-linear): functional A_t extraction from serial fallback")
    print("=" * 70)
    from ndm.models.hybrid_ladder import HybridLadderLM
    for clamp in [False, True]:
        torch.manual_seed(0)
        model = HybridLadderLM(
            vocab_size=task.vocab_size, dim=256, depth=1,
            layer_pattern=['E88'],
            layer_kwargs=[{'linear_state': True, 'use_gate': True,
                           'pos_eigval_clamp': clamp}],
            n_state=32, n_heads=38, expansion=1.0,
        ).to(device).float()
        layer = [m for m in model.modules() if isinstance(m, E88FLAHybrid)][0]
        # embed tokens to get the real layer input the model would feed
        with torch.no_grad():
            h = model.embed(x_tok)
            h = model.layer_norms[0](h)
        ev = alongkey_eig_e88(layer, h.float())
        s = summ(f'e88 clamp={clamp}', ev)
        print(f"  clamp={clamp!s:5}  along-key eig: min={s['min']:+.4f} "
              f"p50={s['p50']:+.4f} max={s['max']:+.4f} frac_neg={s['frac_negative']:.3f}")

    if FLAGatedDeltaNet is not None:
        print("=" * 70)
        print("ARM A  (GDN): exact along-key eigenvalue g*(1-beta) on real inputs")
        print("=" * 70)
        from ndm.models.fla_gated_delta import FLAGatedDeltaNetLayer
        for allow in [False, True]:
            torch.manual_seed(0)
            model = HybridLadderLM(
                vocab_size=task.vocab_size, dim=512, depth=1,
                layer_pattern=['fla-gdn'],
                layer_kwargs=[{'allow_neg_eigval': allow}],
                n_state=32, n_heads=22, expansion=1.0,
            ).to(device).float()
            layer = [m for m in model.modules()
                     if isinstance(m, FLAGatedDeltaNetLayer)][0]
            with torch.no_grad():
                h = model.embed(x_tok)
                h = model.layer_norms[0](h)
            ev = gdn_alongkey(layer, h.float())
            s = summ(f'gdn allow_neg={allow}', ev)
            print(f"  allow_neg={allow!s:5}  along-key eig: min={s['min']:+.4f} "
                  f"p50={s['p50']:+.4f} max={s['max']:+.4f} "
                  f"frac_neg={s['frac_negative']:.3f}")


if __name__ == '__main__':
    main()
