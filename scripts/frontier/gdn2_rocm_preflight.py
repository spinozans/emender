#!/usr/bin/env python3
"""Frontier/ROCm preflight for the external gdn2-MLP path.

Default mode verifies that the production wrapper can import ``fla``, load the
external ``lit_gpt/gdn2.py`` checkout, and find the chunk kernel module. With
``--run-fwdbwd`` it also instantiates a small gdn2-MLP block and runs one
forward/backward step on ``cuda``; on ROCm PyTorch this is the HIP backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_default(value):
    return repr(value)


def _run_fwdbwd(args) -> dict:
    import torch

    from ndm.models.external_gdn2 import GDN2ExternalMLPLayer

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; no HIP/CUDA device for fwdbwd smoke")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    layer = GDN2ExternalMLPLayer(
        dim=args.dim,
        expansion=args.expansion,
        head_dim=args.head_dim,
        num_heads=args.n_heads,
        use_conv=bool(args.use_conv),
        d_conv=args.d_conv,
        gdn2_mlp_ratio=args.mlp_ratio,
    ).to(device=device, dtype=dtype)
    layer.train()

    x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=dtype)
    x.requires_grad_(True)
    y, _ = layer(x)
    loss = y.float().pow(2).mean()
    loss.backward()

    finite_output = bool(torch.isfinite(y.float()).all().item())
    finite_loss = bool(torch.isfinite(loss.detach()).item())
    finite_input_grad = bool(torch.isfinite(x.grad.float()).all().item())
    finite_param_grads = all(
        p.grad is None or bool(torch.isfinite(p.grad.float()).all().item())
        for p in layer.parameters()
    )
    ok = finite_output and finite_loss and finite_input_grad and finite_param_grads
    torch.cuda.synchronize()
    return {
        "ran_fwdbwd": True,
        "device_name": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "dim": args.dim,
            "n_heads": args.n_heads,
            "head_dim": args.head_dim,
            "expansion": args.expansion,
        },
        "loss": float(loss.detach().cpu()),
        "finite_output": finite_output,
        "finite_loss": finite_loss,
        "finite_input_grad": finite_input_grad,
        "finite_param_grads": finite_param_grads,
        "ok": ok,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-fwdbwd", action="store_true", help="launch one gdn2-MLP fwd/bwd smoke")
    parser.add_argument("--bf16", action="store_true", help="use bfloat16 for the fwd/bwd smoke")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--expansion", type=float, default=1.0)
    parser.add_argument("--use-conv", type=int, default=1)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ndm.models.external_gdn2 import probe_gdn2_external_dependencies
    except Exception as exc:
        report = {
            "preflight": {
                "ok": False,
                "failure": f"could not import ndm.models.external_gdn2: {exc!r}",
                "traceback": traceback.format_exc(),
            }
        }
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
        return 1

    report = {"preflight": probe_gdn2_external_dependencies()}
    ok = bool(report["preflight"].get("ok"))

    if ok and args.run_fwdbwd:
        try:
            report["fwdbwd"] = _run_fwdbwd(args)
            ok = bool(report["fwdbwd"].get("ok"))
        except Exception as exc:
            report["fwdbwd"] = {
                "ran_fwdbwd": False,
                "ok": False,
                "failure": repr(exc),
                "traceback": traceback.format_exc(),
            }
            ok = False

    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
