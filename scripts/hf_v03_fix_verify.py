#!/usr/bin/env python3
"""fix-hf-v03 — VERIFY the real fix for the broken HF v0.3 forward.

Established root cause (reconfirmed in this task): the published v0.3
`model.safetensors` are schedule-free *x-mode* (eval-extrapolated) weights, which
are catastrophic at inference (E88 ~18, GDN ~102, M2RNN ~18 nats). The bundled
modeling code (`NdmForCausalLM` + the elman LadderLM/M2RNNLM forward) and config
are FINE — loading the *usable y-mode* weights through them must reproduce the
live-harness reference (E88 2.5598 / GDN 2.5597 / M2RNN 2.5470 nats; bpb 0.9661 /
0.9661 / 0.9613).

This script, GPU 0 only, for each of the three repos:
  A. x-mode contrast: build the genuine bundled NdmForCausalLM from the HF repo,
     load the PUBLISHED v0.3 safetensors strict, measure block-loss + a few
     windows -> expect catastrophic (reproduces the bug through the bundled code).
  B. y-mode fix: load the source training .pt (same step the harness scored, which
     STILL carries optimizer_state_dict), apply the schedule-free y-mode swap
     (optimizer.train()) onto the bundled NdmForCausalLM's inner model, then
     measure the FULL canonical-slice BPB THROUGH THE BUNDLED NdmForCausalLM.forward
     -> expect ~2.56 nats / ~0.966 bpb, matching the harness within <0.01 nats.
  C. stage the corrected repo dir (config + bundled code + tokenizer + the
     re-exported y-mode model.safetensors), reload it via
     AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True), reconfirm
     strict load + a quick gate, and confirm generation works.

NOTHING is pushed. v0.1/v0.2 and the published weights are untouched.
Results -> scripts/hf_v03_fix_verify_result.json (incremental).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # BEFORE torch — never touch GPUs 1-7
os.environ.setdefault("XMA_PATH", "/home/erikg/xma") # M2RNN XMA Triton backend
import sys, glob, json, math, time, shutil, types, importlib, traceback
sys.path.insert(0, "/home/erikg/elman")
sys.path.insert(0, "/home/erikg/elman/elman/cuda")
sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-757/scripts")  # harness helpers

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import measure_pile_bpb_elman as H   # build_model/load_checkpoint_ymode/measure_bpb/block_loss_sanity

LN2 = math.log(2.0)
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"
SLICE_SHA = "3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a"
STAGE_ROOT = "/home/erikg/ndm/.wg-worktrees/agent-757/hf_v03_fix_staging"
OUT = "/home/erikg/ndm/.wg-worktrees/agent-757/scripts/hf_v03_fix_verify_result.json"

MODELS = [
    {"name": "emender-e88-1.3b", "repo": "poietic-pbc/emender-e88-1.3b", "rev": "v0.3",
     "src_pt": "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1542000_loss_2.5970.pt",
     "ref_nats": 2.5597944649733653, "ref_bpb": 0.9661400952828046},
    {"name": "gdn-1.3b", "repo": "poietic-pbc/gdn-1.3b", "rev": "v0.3",
     "src_pt": "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_2031000_loss_2.7303.pt",
     "ref_nats": 2.5597479088882387, "ref_bpb": 0.9661225236765155},
    {"name": "m2rnn-cma-1.3b", "repo": "poietic-pbc/m2rnn-cma-1.3b", "rev": "v0.3",
     "src_pt": "/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1491000_loss_2.7347.pt",
     "ref_nats": 2.547022302196617, "ref_bpb": 0.9613195135013596},
]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def shim_ndm():
    """Map the private `ndm.models.*` names the bundled code imports onto elman."""
    ll = importlib.import_module("elman.models.ladder_lm")
    mb = importlib.import_module("elman.models.m2rnn_baseline")
    ndm = types.ModuleType("ndm"); ndmm = types.ModuleType("ndm.models"); ndm.models = ndmm
    sys.modules.update({"ndm": ndm, "ndm.models": ndmm,
                        "ndm.models.ladder_lm": ll, "ndm.models.m2rnn_baseline": mb})

def snapshot_dir(repo, rev):
    refs = f"/home/erikg/.cache/huggingface/hub/models--{repo.replace('/','--')}/refs/{rev}"
    commit = open(refs).read().strip()
    return f"/home/erikg/.cache/huggingface/hub/models--{repo.replace('/','--')}/snapshots/{commit}"

def build_bundled(spec):
    """Construct the GENUINE bundled NdmForCausalLM (config + dynamic modeling code)."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(spec["repo"], revision=spec["rev"], trust_remote_code=True)
    cls = get_class_from_dynamic_module("modeling_ndm.NdmForCausalLM", spec["repo"], revision=spec["rev"])
    return cfg, cls(cfg)

class HFForward:
    """Adapter so the harness measure_bpb()/block_loss_sanity() call the genuine
    NdmForCausalLM.forward (modeling_ndm.py) and receive raw logits."""
    def __init__(self, hf): self.hf = hf
    def __call__(self, wins):
        return self.hf(wins, return_dict=True).logits
    def eval(self):
        self.hf.eval(); return self

def strip_prefix(sd):
    return {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}

def load_ids():
    import hashlib, tiktoken
    raw = open(SLICE, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == SLICE_SHA, f"slice sha mismatch {sha}"
    text = raw.decode("utf-8")
    assert text.encode("utf-8") == raw, "utf-8 round-trip"
    enc = tiktoken.get_encoding("p50k_base")
    ids = torch.tensor(enc.encode_ordinary(text), dtype=torch.long)[None, :]
    return ids, len(raw), text

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

def save_result(res):
    json.dump(res, open(OUT, "w"), indent=2)

def stage_repo(spec, hf_model):
    """Copy bundled code+config+tokenizer and write the re-exported y-mode safetensors."""
    from safetensors.torch import save_model
    src = snapshot_dir(spec["repo"], spec["rev"])
    dst = os.path.join(STAGE_ROOT, spec["name"])
    os.makedirs(dst, exist_ok=True)
    for fn in ("config.json", "configuration_ndm.py", "modeling_ndm.py",
               "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"):
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(os.path.realpath(s), os.path.join(dst, fn))
    # save_model handles tied weights (shared storage) correctly.
    save_model(hf_model, os.path.join(dst, "model.safetensors"))
    return dst

def verify_reload(spec, dst, ids, device):
    """Reload the staged dir via from_pretrained(trust_remote_code) and gate it."""
    from transformers import AutoModelForCausalLM
    out = {}
    try:
        m = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)
        m = m.to(device).eval()
        out["from_pretrained"] = "OK"
        fwd = HFForward(m)
        out["reload_block_nats"] = float(H.block_loss_sanity(fwd, ids, 2048, device))
        out["reload_5win_nats"] = float(few_window_nats(fwd, ids, device, nwin=5))
        # generation smoke
        try:
            g = m.generate(ids[:, :16].to(device), max_new_tokens=20, do_sample=False)
            out["generation"] = {"ok": True, "out_len": int(g.shape[1])}
        except Exception:
            out["generation"] = {"ok": False, "error": traceback.format_exc().splitlines()[-1]}
        del m; torch.cuda.empty_cache()
    except Exception:
        out["from_pretrained"] = "FAILED"
        out["from_pretrained_error"] = traceback.format_exc()
    return out

def run_model(spec, ids, total_bytes, device, res):
    name = spec["name"]
    r = {"repo": spec["repo"], "rev": spec["rev"], "src_pt": spec["src_pt"],
         "ref_harness_nats": spec["ref_nats"], "ref_harness_bpb": spec["ref_bpb"]}
    # ---- A. x-mode contrast through the bundled code ----
    try:
        cfg, hf = build_bundled(spec)
        snap = os.path.join(snapshot_dir(spec["repo"], spec["rev"]), "model.safetensors")
        miss, unexp = hf.load_state_dict(load_file(snap), strict=False)
        hf = hf.to(device).bfloat16().eval()
        fwd = HFForward(hf)
        xb = float(H.block_loss_sanity(fwd, ids, 2048, device))
        xw = float(few_window_nats(fwd, ids, device, nwin=5))
        r["xmode_published"] = {"missing": len(miss), "unexpected": len(unexp),
                                "block_nats": xb, "few_window_nats": xw,
                                "verdict": "CATASTROPHIC" if xw > 5 else "SANE"}
        log(f"{name} A/x-mode: block={xb:.3f} 5win={xw:.3f} ({r['xmode_published']['verdict']})")
        del hf, fwd; torch.cuda.empty_cache()
    except Exception:
        r["xmode_published"] = {"error": traceback.format_exc()}
        log(f"{name} A/x-mode FAILED:\n" + r["xmode_published"]["error"])
    res[name] = r; save_result(res)

    # ---- B. y-mode fix through the bundled code (FULL slice) ----
    try:
        args_json = json.loads(open(os.path.join(os.path.dirname(spec["src_pt"]), "args.json")).read())
        cfg, hf = build_bundled(spec)
        ckpt = torch.load(spec["src_pt"], map_location="cpu")
        msd = strip_prefix(ckpt["model_state_dict"])
        miss, unexp = hf.model.load_state_dict(msd, strict=False)
        r["ymode_strict_load"] = {"missing": len(miss), "unexpected": len(unexp),
                                  "missing_keys": miss[:8], "unexpected_keys": unexp[:8]}
        log(f"{name} B/y-mode load into bundled .model: missing={len(miss)} unexpected={len(unexp)}")
        hf = hf.to(device).bfloat16()
        # schedule-free y-mode swap (mirrors generate.load_model / measure_pile_bpb_elman)
        if args_json.get("optimizer") == "schedulefree" and "optimizer_state_dict" in ckpt:
            import schedulefree
            opt = schedulefree.AdamWScheduleFree(
                hf.model.parameters(), lr=args_json.get("lr", 3e-4),
                weight_decay=args_json.get("weight_decay", 0.01), betas=(0.9, 0.95))
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            opt.train()
            r["ymode_swap"] = "applied (optimizer.train)"
            log(f"{name} B/y-mode: schedule-free swap applied")
        else:
            r["ymode_swap"] = "NOT APPLIED (not schedulefree or no opt state)"
        hf.tie_weights(); hf.eval()
        fwd = HFForward(hf)
        # gate
        yb = float(H.block_loss_sanity(fwd, ids, 2048, device))
        r["ymode_block_nats"] = yb
        log(f"{name} B/y-mode block-loss gate: {yb:.4f} (expect ~1.6-1.8)")
        if not (1.5 <= yb <= 4.0):
            r["ymode_full"] = {"error": f"gate failed block={yb:.4f}"}
            log(f"{name} B/y-mode GATE FAILED block={yb:.4f} — skipping full slice")
            del ckpt
        else:
            del ckpt
            t0 = time.time()
            m = H.measure_bpb(fwd, ids, 2048, 1024, total_bytes, device, batch_size=8,
                              progress_path=OUT + f".{name}.progress")
            m["seconds"] = round(time.time() - t0, 1)
            m["delta_nats_vs_harness"] = m["nats_per_token"] - spec["ref_nats"]
            m["delta_bpb_vs_harness"] = m["bpb"] - spec["ref_bpb"]
            m["match_within_0.01_nats"] = abs(m["delta_nats_vs_harness"]) < 0.01
            r["ymode_full"] = m
            log(f"{name} B/y-mode FULL: nats={m['nats_per_token']:.4f} bpb={m['bpb']:.4f} "
                f"d_nats={m['delta_nats_vs_harness']:+.4f} ({m['seconds']}s) "
                f"match={m['match_within_0.01_nats']}")
        res[name] = r; save_result(res)
        # ---- C. stage + from_pretrained reload ----
        try:
            dst = stage_repo(spec, hf)
            r["staged_dir"] = dst
            log(f"{name} C: staged -> {dst}")
            del hf, fwd; torch.cuda.empty_cache()
            r["reload"] = verify_reload(spec, dst, ids, device)
            log(f"{name} C: reload {r['reload'].get('from_pretrained')} "
                f"block={r['reload'].get('reload_block_nats')}")
        except Exception:
            r["stage_reload_error"] = traceback.format_exc()
            log(f"{name} C FAILED:\n" + r["stage_reload_error"])
    except Exception:
        r["ymode_full"] = {"error": traceback.format_exc()}
        log(f"{name} B/y-mode FAILED:\n" + r["ymode_full"]["error"])
    res[name] = r; save_result(res)
    torch.cuda.empty_cache()

def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "GPU 0 only"
    shim_ndm()
    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)}")
    ids, total_bytes, _ = load_ids()
    log(f"slice ok: {total_bytes} bytes, {ids.size(1)} tokens")
    res = {"_meta": {"slice_bytes": total_bytes, "tokens": ids.size(1),
                     "context": 2048, "stride": 1024,
                     "note": "y-mode re-export through the genuine bundled NdmForCausalLM.forward"}}
    save_result(res)
    for spec in MODELS:
        log(f"================= {spec['name']} =================")
        try:
            run_model(spec, ids, total_bytes, device, res)
        except Exception:
            res[spec["name"]] = {"fatal": traceback.format_exc()}
            save_result(res)
            log(f"{spec['name']} FATAL:\n" + res[spec['name']]["fatal"])
    log("ALL DONE")
    save_result(res)

if __name__ == "__main__":
    main()
