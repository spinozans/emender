#!/usr/bin/env python3
"""Diagnose the CUDA device-side assert in the HF v0.3 E88 forward.

Runs with CUDA_LAUNCH_BLOCKING=1 so the assert points at the real op.
Checks: token-id range vs vocab, tiny-input forward, use_triton on/off,
dtype fp32 fallback. Prints a precise root-cause signal.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import sys, json, glob, traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

REPO = "poietic-pbc/emender-e88-1.3b"
REV = "v0.3"
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"


def banner(s): print(f"\n===== {s} =====", flush=True)


def main():
    banner("env")
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0), flush=True)

    banner("config")
    cfg = AutoConfig.from_pretrained(REPO, revision=REV, trust_remote_code=True)
    for k in ["vocab_size","dim","depth","n_heads","n_state","n_groups","n_slots",
              "expansion","state_expansion","level","e88_decay_mode","gate_activation",
              "linear_state","use_gate","use_write_gate","use_conv","d_conv",
              "use_triton","r_h_mode","top_k","k_fast","k_slow","checkpoint_interval"]:
        print(f"  {k} = {getattr(cfg, k, '<MISSING>')}", flush=True)

    banner("tokenizer / id range")
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV, trust_remote_code=True)
    with open(SLICE, "rb") as f:
        data = f.read()
    text = data.decode("utf-8")
    ids = tok(text[:20000], add_special_tokens=False)["input_ids"]
    import builtins
    print("  n sample tokens", len(ids), "min", builtins.min(ids), "max", builtins.max(ids),
          "vocab_size", cfg.vocab_size, flush=True)
    over = [i for i in ids if i >= cfg.vocab_size]
    print("  tokens >= vocab_size:", len(over), over[:10], flush=True)

    def try_forward(use_triton, dtype, ntok):
        label = f"use_triton={use_triton} dtype={dtype} ntok={ntok}"
        banner(f"forward {label}")
        try:
            c = AutoConfig.from_pretrained(REPO, revision=REV, trust_remote_code=True)
            c.use_triton = use_triton
            model = AutoModelForCausalLM.from_pretrained(
                REPO, revision=REV, trust_remote_code=True, config=c, torch_dtype=dtype)
            model.to("cuda").eval()
            x = torch.tensor([ids[:ntok]], device="cuda")
            with torch.no_grad():
                out = model(x)
            lg = out.logits
            print(f"  OK logits {tuple(lg.shape)} finite={torch.isfinite(lg).all().item()} "
                  f"dtype={lg.dtype}", flush=True)
            # quick nats/token on this tiny window
            import torch.nn.functional as F
            nll = F.cross_entropy(lg[0,:-1].float(), x[0,1:], reduction="mean")
            print(f"  tiny-window mean nats/token = {nll.item():.4f}", flush=True)
            del model; torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            traceback.print_exc()
            try:
                del model; torch.cuda.empty_cache()
            except Exception:
                pass
            return False

    # Order: cheap + most likely informative first.
    cfg_triton = getattr(cfg, "use_triton", None)
    try_forward(use_triton=False, dtype=torch.float32, ntok=16)
    try_forward(use_triton=False, dtype=torch.bfloat16, ntok=2048)
    if cfg_triton:
        try_forward(use_triton=True, dtype=torch.bfloat16, ntok=16)
    print("\nDIAG DONE", flush=True)


if __name__ == "__main__":
    main()
