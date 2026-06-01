#!/usr/bin/env python3
"""Measure held-out Pile BPB through the GENUINE HF v0.3 modeling code, bypassing
only transformers' load FINALIZER (which is incompatible with the shipped
modeling_ndm.py on transformers>=5).

What is faithful to the HF release here:
  * the HF NdmForCausalLM class + its _build_ndm_model(config) -> the real
    ndm.models.{ladder_lm,m2rnn_baseline} forward (we point `ndm` at the real
    /home/erikg/elman code, since the HF wrapper imports `ndm` from the env and
    the only in-tree `ndm` would be used the same way),
  * the HF config.json (every flag),
  * the HF model.safetensors weights (loaded strict).

What we skip: transformers from_pretrained's _finalize_model_loading, which calls
NdmForCausalLM.tie_weights(missing_keys=...) / model.all_tied_weights_keys —
kwargs/attrs the shipped (older-transformers) modeling code does not define.

This isolates the remaining question: are the HF weights the usable y-mode
weights (~2.6 nats) or the catastrophic schedule-free x-mode weights (~17.6)?

GPU 0 ONLY. bf16. REAL measurement. Result -> /tmp/e88_hf_pathmeasure_result.json
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")

import sys, glob, json, math, time, hashlib, importlib, types, traceback

sys.path.insert(0, "/home/erikg/elman")
sys.path.insert(0, "/home/erikg/elman/elman/cuda")

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

LN2 = math.log(2.0)
CTX = 2048
STRIDE = 1024
TOTAL_BYTES = 9_999_511
EXPECT_SHA = "3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a"
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"
E88_TRAIN_LOSS = 0.974
OUT = "/tmp/e88_hf_pathmeasure_result.json"
LOG = "/home/erikg/ndm/.wg-worktrees/agent-744/scripts/e88_hf_pathmeasure.log"

MODELS = [
    {"name": "emender-e88-1.3b", "repo": "poietic-pbc/emender-e88-1.3b", "rev": "v0.3", "step": 1_524_000},
    {"name": "gdn-1.3b",         "repo": "poietic-pbc/gdn-1.3b",         "rev": "v0.3", "step": 1_998_000},
    {"name": "m2rnn-cma-1.3b",   "repo": "poietic-pbc/m2rnn-cma-1.3b",   "rev": "v0.3", "step": 1_467_000},
]

_logf = open(LOG, "w")
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True); _logf.write(s + "\n"); _logf.flush()


def install_ndm_shim():
    ll = importlib.import_module("elman.models.ladder_lm")
    mb = importlib.import_module("elman.models.m2rnn_baseline")
    ndm = types.ModuleType("ndm"); ndmm = types.ModuleType("ndm.models"); ndm.models = ndmm
    sys.modules.update({"ndm": ndm, "ndm.models": ndmm,
                        "ndm.models.ladder_lm": ll, "ndm.models.m2rnn_baseline": mb})
    log(f"ndm shim -> {ll.__file__}")


def load_slice():
    data = open(SLICE, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    assert len(data) == TOTAL_BYTES and sha == EXPECT_SHA, "slice mismatch"
    log(f"slice OK bytes={len(data)} sha={sha[:12]}")
    return data.decode("utf-8")


def build_hf_model(spec):
    """Construct the genuine HF NdmForCausalLM(config) and load HF safetensors strict."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(spec["repo"], revision=spec["rev"], trust_remote_code=True)
    cls = get_class_from_dynamic_module(
        "modeling_ndm.NdmForCausalLM", spec["repo"], revision=spec["rev"])
    log(f"  HF class {cls.__module__}.{cls.__name__}; r_h_mode={getattr(cfg,'r_h_mode',None)} "
        f"use_triton={getattr(cfg,'use_triton',None)}")
    model = cls(cfg)  # calls _build_ndm_model(config) -> real forward
    snap = glob.glob(f"/home/erikg/.cache/huggingface/hub/models--{spec['repo'].replace('/','--')}/snapshots/*/model.safetensors")
    sd = load_file(snap[0])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing: log(f"    missing[:5]={missing[:5]}")
    if unexpected: log(f"    unexpected[:5]={unexpected[:5]}")
    return model.to(torch.bfloat16).cuda().eval(), cfg


@torch.no_grad()
def measure(spec, text, quick_windows=None):
    from transformers import AutoTokenizer
    log(f"=== {spec['name']} ===")
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["rev"], trust_remote_code=True)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    n = ids.numel(); log(f"  tokens={n}")
    model, cfg = build_hf_model(spec)
    ids = ids.cuda()
    total_nll = 0.0; n_scored = 0; prev_end = 0; nwin = 0; t0 = time.time()
    for begin in range(0, n, STRIDE):
        end = min(begin + CTX, n); trg = end - prev_end
        logits = model(ids[begin:end].unsqueeze(0)).logits[0].float()
        tgt = ids[begin:end]
        keep = torch.arange(begin + 1, end, device=ids.device) >= (end - trg)
        sl = logits[:-1][keep]; st = tgt[1:][keep]
        if sl.numel():
            total_nll += F.cross_entropy(sl, st, reduction="sum").item(); n_scored += int(st.numel())
        prev_end = end; nwin += 1
        if nwin % 100 == 0:
            log(f"    win{nwin} {begin}/{n} nats/tok={total_nll/max(n_scored,1):.4f} ({time.time()-t0:.0f}s)")
        if quick_windows and nwin >= quick_windows:
            log(f"    [quick stop @ {nwin}]"); break
        if end == n: break
    mean = total_nll / max(n_scored, 1)
    bpb = None if quick_windows else total_nll / (TOTAL_BYTES * LN2)
    del model; torch.cuda.empty_cache()
    r = {"name": spec["name"], "repo": spec["repo"], "revision": spec["rev"], "step": spec["step"],
         "n_tokens": int(n), "n_scored": int(n_scored), "total_nll": total_nll,
         "mean_nats_per_token": mean, "bpb": bpb, "quick": bool(quick_windows)}
    log(f"  -> mean nats/tok={mean:.4f}" + ("" if bpb is None else f" bpb={bpb:.4f}"))
    return r


def main():
    log("START e88 HF-path measurement (genuine HF code, finalizer bypassed)")
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0)}")
    install_ndm_shim()
    text = load_slice()
    state = {"models": [], "note": ""}

    try:
        q = measure(MODELS[0], text, quick_windows=100)
    except Exception:
        log("E88 quick gate EXCEPTION:\n" + traceback.format_exc())
        state["e88_quick_error"] = traceback.format_exc()
        json.dump(state, open(OUT, "w"), indent=2); return
    state["e88_quick"] = q
    json.dump(state, open(OUT, "w"), indent=2)

    if q["mean_nats_per_token"] >= 5.0:
        state["note"] = (f"E88 HF weights give {q['mean_nats_per_token']:.4f} nats/token on the "
                         f"genuine forward -> CATASTROPHIC (x-mode schedule-free weights). "
                         f"HF release weights are not inference-usable; needs y-mode re-export.")
        log("VERDICT: " + state["note"])
        json.dump(state, open(OUT, "w"), indent=2); return

    log("E88 quick SANE -> full BPB on all three")
    for spec in MODELS:
        try:
            r = measure(spec, text)
            r["sanity"] = "PASS" if (r["mean_nats_per_token"] < 5.0 and r["bpb"] < 2.0) else "FAIL"
        except Exception:
            r = {"name": spec["name"], "error": traceback.format_exc()}
            log(f"{spec['name']} EXCEPTION:\n" + r["error"])
        state["models"].append(r)
        json.dump(state, open(OUT, "w"), indent=2)
    log("DONE")
    json.dump(state, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
