#!/usr/bin/env python3
"""Reload each corrected staged dir via AutoModelForCausalLM.from_pretrained(
trust_remote_code=True), confirm a sane forward (block-loss + 5 windows on the
canonical slice) and that generation runs. GPU 0 only. Appends `reload` per repo
to scripts/hf_v03_fix_verify_result.json."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys, json, time, types, importlib, traceback
sys.path.insert(0, "/home/erikg/elman")
sys.path.insert(0, "/home/erikg/elman/elman/cuda")
sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-757/scripts")
import torch, torch.nn.functional as F
import measure_pile_bpb_elman as H

OUT = "/home/erikg/ndm/.wg-worktrees/agent-757/scripts/hf_v03_fix_verify_result.json"
STAGE_ROOT = "/home/erikg/ndm/.wg-worktrees/agent-757/hf_v03_fix_staging"
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"
SLICE_SHA = "3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a"
ALL = ["emender-e88-1.3b", "gdn-1.3b", "m2rnn-cma-1.3b"]
REF = {"emender-e88-1.3b": 2.5597944649733653, "gdn-1.3b": 2.5597479088882387,
       "m2rnn-cma-1.3b": 2.547022302196617}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def shim():
    ll = importlib.import_module("elman.models.ladder_lm")
    mb = importlib.import_module("elman.models.m2rnn_baseline")
    ndm = types.ModuleType("ndm"); ndmm = types.ModuleType("ndm.models"); ndm.models = ndmm
    sys.modules.update({"ndm": ndm, "ndm.models": ndmm,
                        "ndm.models.ladder_lm": ll, "ndm.models.m2rnn_baseline": mb})

def load_ids():
    import hashlib, tiktoken
    raw = open(SLICE, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == SLICE_SHA
    text = raw.decode("utf-8"); assert text.encode("utf-8") == raw
    enc = tiktoken.get_encoding("p50k_base")
    return torch.tensor(enc.encode_ordinary(text), dtype=torch.long)[None, :]

class HFForward:
    def __init__(self, hf): self.hf = hf
    def __call__(self, wins): return self.hf(wins, return_dict=True).logits

@torch.no_grad()
def few_window_nats(fwd, ids, device, nwin=5):
    total = 0.0; n = 0; prev = 0; w = 0
    for begin in range(0, ids.size(1), 1024):
        end = min(begin + 2048, ids.size(1)); trg = end - prev
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(ids[:, begin:end].to(device))[0].float()
        tgt = ids[0, begin:end].to(device)
        keep = torch.arange(begin + 1, end, device=device) >= (end - trg)
        total += F.cross_entropy(lg[:-1][keep], tgt[1:][keep], reduction="sum").item()
        n += int((tgt[1:][keep]).numel()); prev = end; w += 1
        if w >= nwin or end == ids.size(1): break
    return total / max(n, 1)

def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    from transformers import AutoModelForCausalLM
    shim()
    device = torch.device("cuda")
    ids = load_ids()
    res = json.load(open(OUT))
    for name in ALL:
        dst = os.path.join(STAGE_ROOT, name)
        if not os.path.isdir(dst):
            log(f"{name}: no staged dir"); continue
        log(f"===== reload: {name} =====")
        out = {}
        try:
            t0 = time.time()
            m = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True, dtype=torch.bfloat16)
            m = m.to(device).eval()
            out["from_pretrained"] = "OK"
            out["load_seconds"] = round(time.time() - t0, 1)
            fwd = HFForward(m)
            out["reload_block_nats"] = float(H.block_loss_sanity(fwd, ids, 2048, device))
            out["reload_5win_nats"] = float(few_window_nats(fwd, ids, device, 5))
            out["ref_harness_nats"] = REF[name]
            out["sane"] = bool(out["reload_block_nats"] < 4.0 and out["reload_5win_nats"] < 4.0)
            try:
                g = m.generate(ids[:, :16].to(device), max_new_tokens=20, do_sample=False)
                out["generation"] = {"ok": int(g.shape[1]) == 36, "out_len": int(g.shape[1])}
            except Exception:
                out["generation"] = {"ok": False, "error": traceback.format_exc().splitlines()[-1]}
            log(f"{name}: OK block={out['reload_block_nats']:.4f} 5win={out['reload_5win_nats']:.4f} "
                f"sane={out['sane']} gen={out['generation']}")
            del m, fwd; torch.cuda.empty_cache()
        except Exception:
            out["from_pretrained"] = "FAILED"; out["error"] = traceback.format_exc()
            log(f"{name}: FAILED\n" + out["error"])
        res.setdefault(name, {})["reload"] = out
        json.dump(res, open(OUT, "w"), indent=2)
    log("RELOAD DONE")

if __name__ == "__main__":
    main()
