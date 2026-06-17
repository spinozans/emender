#!/usr/bin/env python3
"""Confound check for NON-NEGOTIABLE #1: does the emender (E97) recurrence run
through the FUSED Triton kernel at EVAL time (the capability/BPB harness setting),
or does it silently fall back to the EAGER per-step PyTorch loop?

Loads the REAL reference checkpoint via the exact eval-harness loader, instruments
the fused-kernel entry points and the eager per-step activation, and runs a
forward under (a) the harness setting model.eval()+no_grad and (b) model.train(),
counting which path executes. Also validates the optional fused-inference fix
(set mixer.fused_inference=True) if --fix is given.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import eval_checkpoint as ckpt_mod  # noqa: E402
import ndm.triton.e88_triton_optimized as topt  # noqa: E402
import ndm.triton.e97_chunked_autograd as tchunk  # noqa: E402
from ndm.models.e88_fla_hybrid import E88FLAHybrid  # noqa: E402

COUNTS = {"fused_seq": 0, "fused_chunked": 0, "eager_act": 0}

_orig_seq = topt.e88_triton_optimized_apply
_orig_chunk = tchunk.e97_delta_chunked_triton
_orig_act = E88FLAHybrid._apply_state_activation

def _seq(*a, **k):
    COUNTS["fused_seq"] += 1
    return _orig_seq(*a, **k)
def _chunk(*a, **k):
    COUNTS["fused_chunked"] += 1
    return _orig_chunk(*a, **k)
def _act(self, pre):
    COUNTS["eager_act"] += 1
    return _orig_act(self, pre)

topt.e88_triton_optimized_apply = _seq
tchunk.e97_delta_chunked_triton = _chunk
E88FLAHybrid._apply_state_activation = _act


def reset():
    for kx in COUNTS:
        COUNTS[kx] = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--fix", action="store_true",
                    help="set mixer.fused_inference=True before the eval forward")
    args = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt_mod.checkpoint_args(Path(args.ckpt), ck, None)
    margs = ckpt_mod.namespace_from_config(cfg)
    model = ckpt_mod.build_model(margs, dev)
    swapped = ckpt_mod.load_checkpoint_weights(model, ck, margs, "train")
    n_mixers = sum(isinstance(m, E88FLAHybrid) for m in model.modules())
    print(f"level={margs.level} use_triton={getattr(margs,'use_triton',None)} "
          f"linear_state={getattr(margs,'linear_state',None)} sf_y_swap={swapped} "
          f"E88FLAHybrid_mixers={n_mixers}")
    x = torch.randint(0, 256, (1, args.seqlen), device=dev)

    # (a) harness setting: model.eval() + no_grad
    model.eval()
    if args.fix:
        n_set = 0
        for m in model.modules():
            if isinstance(m, E88FLAHybrid):
                m.fused_inference = True
                n_set += 1
        print(f"[fix] set fused_inference=True on {n_set} mixers")
    reset()
    with torch.no_grad():
        model(x, return_loss=False)
    print(f"EVAL  : fused_seq={COUNTS['fused_seq']} fused_chunked={COUNTS['fused_chunked']} "
          f"eager_act={COUNTS['eager_act']}")
    eval_counts = dict(COUNTS)

    # (b) train() forward (proves the fused kernel IS reachable)
    model.train()
    reset()
    with torch.no_grad():
        model(x, return_loss=False)
    print(f"TRAIN : fused_seq={COUNTS['fused_seq']} fused_chunked={COUNTS['fused_chunked']} "
          f"eager_act={COUNTS['eager_act']}")

    fused_eval = eval_counts["fused_seq"] + eval_counts["fused_chunked"]
    print("VERDICT: " + ("FUSED at eval (PASS)" if fused_eval > 0 and eval_counts["eager_act"] == 0
                          else "EAGER at eval (NON-NEGOTIABLE #1 VIOLATION)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
