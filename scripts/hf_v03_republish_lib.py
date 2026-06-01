#!/usr/bin/env python3
"""Shared helpers for republish-hf-v03 — y-mode weights overwrite on HF @v0.3.

GPU 0 only. REAL data only. This module is imported by the build/gate, the
upload, and the post-upload readback scripts so the SAME canonical slice and the
SAME genuine-bundled forward path are used everywhere.

The readback loads the GENUINE published modeling code (the dynamic
`modeling_ndm.NdmForCausalLM` shipped on @v0.3) and the GENUINE published
config; the only environment shim is mapping the private `ndm.models.*` names
the bundled code imports onto the installed `elman` package (the documented
local-verification shim from HF_V03_FIX.md — it is NOT a repo file change).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # BEFORE torch — GPU 0 only
os.environ.setdefault("XMA_PATH", "/home/erikg/xma") # M2RNN XMA Triton backend
import sys, math, time, types, importlib, hashlib

sys.path.insert(0, "/home/erikg/elman")
sys.path.insert(0, "/home/erikg/elman/elman/cuda")
sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-757/scripts")  # measure_pile_bpb_elman

import torch
import torch.nn.functional as F

LN2 = math.log(2.0)
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"
SLICE_SHA = "3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a"

# Verified y-mode staging (full-slice verified by fix-hf-v03 / hf_v03_fix_verify.py)
STAGE_ROOT = "/home/erikg/ndm/.wg-worktrees/agent-757/hf_v03_fix_staging"

MODELS = [
    {"key": "e88", "name": "emender-e88-1.3b", "repo": "poietic-pbc/emender-e88-1.3b",
     "identity": "Emender/E88",
     "src_pt": "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1542000_loss_2.5970.pt",
     "ckpt_step": 1542000, "ref_nats": 2.5597944649733653, "ref_bpb": 0.9661400952828046},
    {"key": "gdn", "name": "gdn-1.3b", "repo": "poietic-pbc/gdn-1.3b",
     "identity": "GDN",
     "src_pt": "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_2031000_loss_2.7303.pt",
     "ckpt_step": 2031000, "ref_nats": 2.5597479088882387, "ref_bpb": 0.9661225236765155},
    {"key": "m2rnn", "name": "m2rnn-cma-1.3b", "repo": "poietic-pbc/m2rnn-cma-1.3b",
     "identity": "M2RNN-CMA",
     "src_pt": "/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1491000_loss_2.7347.pt",
     "ckpt_step": 1491000, "ref_nats": 2.547022302196617, "ref_bpb": 0.9613195135013596},
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def shim_ndm():
    """Map the private `ndm.models.*` names the bundled code imports onto elman.
    Runtime-only environment shim (NOT a repo change) — see HF_V03_FIX.md."""
    ll = importlib.import_module("elman.models.ladder_lm")
    mb = importlib.import_module("elman.models.m2rnn_baseline")
    ndm = types.ModuleType("ndm"); ndmm = types.ModuleType("ndm.models"); ndm.models = ndmm
    sys.modules.update({"ndm": ndm, "ndm.models": ndmm,
                        "ndm.models.ladder_lm": ll, "ndm.models.m2rnn_baseline": mb})


def load_ids():
    import tiktoken
    raw = open(SLICE, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == SLICE_SHA, f"slice sha mismatch {sha}"
    text = raw.decode("utf-8")
    assert text.encode("utf-8") == raw, "utf-8 round-trip"
    enc = tiktoken.get_encoding("p50k_base")
    ids = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)[None, :]
    return ids, len(raw)


class HFForward:
    """Adapter so the harness measure_bpb()/block_loss_sanity() call the genuine
    NdmForCausalLM.forward (modeling_ndm.py) and receive raw logits."""
    def __init__(self, hf): self.hf = hf
    def __call__(self, wins):
        return self.hf(wins, return_dict=True).logits
    def eval(self):
        self.hf.eval(); return self


@torch.no_grad()
def few_window_nats(fwd, ids, device, nwin=5, context=2048, stride=1024):
    total = 0.0; n = 0; prev = 0; w = 0
    for begin in range(0, ids.size(1), stride):
        end = min(begin + context, ids.size(1)); trg = end - prev
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(ids[:, begin:end].to(device))[0].float()
        tgt = ids[0, begin:end].to(device)
        keep = torch.arange(begin + 1, end, device=device) >= (end - trg)
        sl = lg[:-1][keep]; st = tgt[1:][keep]
        total += F.cross_entropy(sl, st, reduction="sum").item(); n += int(st.numel())
        prev = end; w += 1
        if w >= nwin or end == ids.size(1): break
    return total / max(n, 1)
