"""Numerical witnesses for one-step recurrent transition separations.

This is not a language-model training benchmark. It directly tests the update
families discussed in the Lean paper core:

* NDM target: tanh((I - k k^T) H + k v^T)
* GDN delta core: a learned affine rescaling of (I - k k^T) H + k v^T
* M2RNN-style candidate: f_k H + (1 - f_k) tanh(H W + k v^T)

The point is to make the formal local separation visible numerically:

* M2RNN has trouble because W is fixed and applied on the right.
* GDN has the correct delta preactivation but lacks recurrent nonlinearity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def basis_keys(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    idx = torch.randint(0, dim, (batch,), device=device)
    return F.one_hot(idx, num_classes=dim).float()


def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, :, None] * b[:, None, :]


def ndm_preactivation(H: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      lam: float = 1.0) -> torch.Tensor:
    batch, dim, _ = H.shape
    eye = torch.eye(dim, device=H.device).expand(batch, dim, dim)
    transition = lam * eye - outer(k, k)
    return torch.bmm(transition, H) + outer(k, v)


def ndm_target(H: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               lam: float = 1.0) -> torch.Tensor:
    return torch.tanh(ndm_preactivation(H, k, v, lam=lam))


def train_gdn_surrogate(H: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        target: torch.Tensor, steps: int, lr: float) -> tuple[float, dict]:
    """Best small affine wrapper around the GDN delta preactivation."""
    scale = torch.nn.Parameter(torch.tensor(1.0, device=H.device))
    bias = torch.nn.Parameter(torch.tensor(0.0, device=H.device))
    opt = torch.optim.AdamW([scale, bias], lr=lr, weight_decay=0.0)
    pre = ndm_preactivation(H, k, v).detach()
    for _ in range(steps):
        pred = scale * pre + bias
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(F.mse_loss(scale * pre + bias, target).item()), {
        "scale": float(scale.item()),
        "bias": float(bias.item()),
    }


def train_m2rnn_surrogate(H: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          target: torch.Tensor, steps: int, lr: float) -> tuple[float, dict]:
    """Fit fixed-right-transition M2RNN with key-dependent scalar forget."""
    _, dim, _ = H.shape
    W = torch.nn.Parameter(torch.eye(dim, device=H.device) + 0.01 * torch.randn(dim, dim, device=H.device))
    forget_logits = torch.nn.Parameter(torch.zeros(dim, device=H.device))
    opt = torch.optim.AdamW([W, forget_logits], lr=lr, weight_decay=0.0)

    key_index = k.argmax(dim=-1)
    raw_write = outer(k, v)
    for _ in range(steps):
        f = torch.sigmoid(forget_logits[key_index])[:, None, None]
        candidate = torch.tanh(torch.matmul(H, W) + raw_write)
        pred = f * H + (1.0 - f) * candidate
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        f = torch.sigmoid(forget_logits[key_index])[:, None, None]
        candidate = torch.tanh(torch.matmul(H, W) + raw_write)
        pred = f * H + (1.0 - f) * candidate
        loss = F.mse_loss(pred, target)
    return float(loss.item()), {
        "forget": [float(x) for x in torch.sigmoid(forget_logits).detach().cpu()],
        "W": [[float(x) for x in row] for row in W.detach().cpu()],
    }


def run(args: argparse.Namespace) -> list[dict]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    results = []

    for state_scale in args.state_scales:
        H = state_scale * torch.randn(args.batch, args.dim, args.dim, device=device)
        v = args.value_scale * torch.randn(args.batch, args.dim, device=device)
        k = basis_keys(args.batch, args.dim, device)
        target = ndm_target(H, k, v)
        pre = ndm_preactivation(H, k, v)

        ndm_mse = float(F.mse_loss(target, target).item())
        raw_gdn_mse = float(F.mse_loss(pre, target).item())
        gdn_mse, gdn_params = train_gdn_surrogate(H, k, v, target, args.steps, args.lr)
        m2_mse, m2_params = train_m2rnn_surrogate(H, k, v, target, args.steps, args.lr)

        row = {
            "dim": args.dim,
            "batch": args.batch,
            "state_scale": state_scale,
            "value_scale": args.value_scale,
            "ndm_oracle_mse": ndm_mse,
            "gdn_raw_delta_mse": raw_gdn_mse,
            "gdn_affine_delta_mse": gdn_mse,
            "m2rnn_fixed_right_mse": m2_mse,
            "gdn_params": gdn_params,
            "m2_params": m2_params,
        }
        results.append(row)
        print(
            f"scale={state_scale:<5g}  "
            f"NDM={ndm_mse:.3e}  "
            f"GDN(raw)={raw_gdn_mse:.3e}  "
            f"GDN(affine)={gdn_mse:.3e}  "
            f"M2={m2_mse:.3e}",
            flush=True,
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {out}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--state_scales", nargs="+", type=float,
                        default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--value_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--output", default="experiments/expressivity_tasks/results/transition_witness.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
