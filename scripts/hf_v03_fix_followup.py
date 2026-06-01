#!/usr/bin/env python3
"""fix-hf-v03 follow-up: (1) re-run GDN y-mode full-slice BPB through the bundled
NdmForCausalLM (the first run OOM'd under transient GPU-0 contention; smaller batch
+ expandable segments here); (2) fix the staged safetensors `format` metadata and
apply the transformers-robustness + vendored-import packaging patch to each staged
`modeling_ndm.py`; (3) verify each staged dir loads via
AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True) and generates.

GPU 0 only. Appends to scripts/hf_v03_fix_verify_result.json (the GDN ymode_full
and per-repo `reload` entries).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys, glob, json, math, time, types, importlib, traceback, re
sys.path.insert(0, "/home/erikg/elman")
sys.path.insert(0, "/home/erikg/elman/elman/cuda")
sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-757/scripts")
import torch, torch.nn.functional as F
from safetensors.torch import load_file, save_file, save_model
import measure_pile_bpb_elman as H

OUT = "/home/erikg/ndm/.wg-worktrees/agent-757/scripts/hf_v03_fix_verify_result.json"
STAGE_ROOT = "/home/erikg/ndm/.wg-worktrees/agent-757/hf_v03_fix_staging"
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"
SLICE_SHA = "3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a"

GDN = {"name": "gdn-1.3b", "repo": "poietic-pbc/gdn-1.3b", "rev": "v0.3",
       "src_pt": "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_2031000_loss_2.7303.pt",
       "ref_nats": 2.5597479088882387, "ref_bpb": 0.9661225236765155}
ALL = ["emender-e88-1.3b", "gdn-1.3b", "m2rnn-cma-1.3b"]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def shim_ndm():
    ll = importlib.import_module("elman.models.ladder_lm")
    mb = importlib.import_module("elman.models.m2rnn_baseline")
    ndm = types.ModuleType("ndm"); ndmm = types.ModuleType("ndm.models"); ndm.models = ndmm
    sys.modules.update({"ndm": ndm, "ndm.models": ndmm,
                        "ndm.models.ladder_lm": ll, "ndm.models.m2rnn_baseline": mb})

def snap_dir(repo, rev):
    commit = open(f"/home/erikg/.cache/huggingface/hub/models--{repo.replace('/','--')}/refs/{rev}").read().strip()
    return f"/home/erikg/.cache/huggingface/hub/models--{repo.replace('/','--')}/snapshots/{commit}"

def build_bundled(spec):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(spec["repo"], revision=spec["rev"], trust_remote_code=True)
    cls = get_class_from_dynamic_module("modeling_ndm.NdmForCausalLM", spec["repo"], revision=spec["rev"])
    return cfg, cls(cfg)

class HFForward:
    def __init__(self, hf): self.hf = hf
    def __call__(self, wins): return self.hf(wins, return_dict=True).logits
    def eval(self): self.hf.eval(); return self

def strip_prefix(sd):
    return {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}

def load_ids():
    import hashlib, tiktoken
    raw = open(SLICE, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == SLICE_SHA
    text = raw.decode("utf-8"); assert text.encode("utf-8") == raw
    enc = tiktoken.get_encoding("p50k_base")
    return torch.tensor(enc.encode_ordinary(text), dtype=torch.long)[None, :], len(raw)

def load_res(): return json.load(open(OUT))
def save_res(r): json.dump(r, open(OUT, "w"), indent=2)

PATCH_HEADER = "# [fix-hf-v03] packaging patch: vendored-import fallback + transformers-robust tying\n"

def patch_modeling(dst):
    """Make the staged modeling_ndm.py self-contained on a clean machine + robust to
    transformers>=5 loader (`all_tied_weights_keys`, `tie_weights(*args,**kwargs)`)."""
    p = os.path.join(dst, "modeling_ndm.py")
    src = open(p).read()
    if PATCH_HEADER in src:
        return "already patched"
    # 1. ndm.* import fallback to elman.*
    src = src.replace(
        'module = importlib.import_module("ndm.models.m2rnn_baseline")',
        'try:\n        module = importlib.import_module("ndm.models.m2rnn_baseline")\n'
        '    except ModuleNotFoundError:\n        module = importlib.import_module("elman.models.m2rnn_baseline")')
    src = src.replace(
        'module = importlib.import_module("ndm.models.ladder_lm")',
        'try:\n        module = importlib.import_module("ndm.models.ladder_lm")\n'
        '    except ModuleNotFoundError:\n        module = importlib.import_module("elman.models.ladder_lm")')
    # 2. transformers-robust tie_weights signature
    src = src.replace("    def tie_weights(self):\n",
                      "    def tie_weights(self, *args, **kwargs):\n")
    # 3. all_tied_weights_keys attribute (loader on transformers 5.x reads it)
    src = src.replace('    _tied_weights_keys = ["model.lm_head.weight"]\n',
                      '    _tied_weights_keys = ["model.lm_head.weight"]\n'
                      '    all_tied_weights_keys = ["model.lm_head.weight"]\n')
    open(p, "w").write(PATCH_HEADER + src)
    return "patched"

def fix_safetensors_metadata(dst):
    """Rewrite model.safetensors with `format: pt` metadata (transformers requires it)."""
    p = os.path.join(dst, "model.safetensors")
    sd = load_file(p)
    save_file(sd, p, metadata={"format": "pt"})
    return len(sd)

def stage_gdn(hf_model):
    import shutil
    src = snap_dir(GDN["repo"], GDN["rev"]); dst = os.path.join(STAGE_ROOT, GDN["name"])
    os.makedirs(dst, exist_ok=True)
    for fn in ("config.json", "configuration_ndm.py", "modeling_ndm.py",
               "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json"):
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(os.path.realpath(s), os.path.join(dst, fn))
    save_model(hf_model, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    return dst

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

def reload_check(name, ids, device):
    from transformers import AutoModelForCausalLM
    dst = os.path.join(STAGE_ROOT, name)
    out = {}
    try:
        m = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True, torch_dtype=torch.bfloat16)
        m = m.to(device).eval()
        fwd = HFForward(m)
        out["from_pretrained"] = "OK"
        out["reload_block_nats"] = float(H.block_loss_sanity(fwd, ids, 2048, device))
        out["reload_5win_nats"] = float(few_window_nats(fwd, ids, device, 5))
        try:
            g = m.generate(ids[:, :16].to(device), max_new_tokens=20, do_sample=False)
            txt_ok = int(g.shape[1]) == 36
            out["generation"] = {"ok": bool(txt_ok), "out_len": int(g.shape[1])}
        except Exception:
            out["generation"] = {"ok": False, "error": traceback.format_exc().splitlines()[-1]}
        del m; torch.cuda.empty_cache()
    except Exception:
        out["from_pretrained"] = "FAILED"; out["error"] = traceback.format_exc()
    return out

def rerun_gdn(ids, total_bytes, device, res):
    spec = GDN; name = spec["name"]
    r = res.get(name, {})
    try:
        args_json = json.loads(open(os.path.join(os.path.dirname(spec["src_pt"]), "args.json")).read())
        cfg, hf = build_bundled(spec)
        ckpt = torch.load(spec["src_pt"], map_location="cpu")
        miss, unexp = hf.model.load_state_dict(strip_prefix(ckpt["model_state_dict"]), strict=False)
        r["ymode_strict_load"] = {"missing": len(miss), "unexpected": len(unexp)}
        log(f"GDN strict load missing={len(miss)} unexpected={len(unexp)}")
        hf = hf.to(device).bfloat16()
        import schedulefree
        opt = schedulefree.AdamWScheduleFree(hf.model.parameters(), lr=args_json.get("lr", 3e-4),
                                             weight_decay=args_json.get("weight_decay", 0.01), betas=(0.9, 0.95))
        opt.load_state_dict(ckpt["optimizer_state_dict"]); opt.train(); del ckpt
        hf.tie_weights(); hf.eval()
        fwd = HFForward(hf)
        yb = float(H.block_loss_sanity(fwd, ids, 2048, device))
        r["ymode_block_nats"] = yb; log(f"GDN y-mode block gate {yb:.4f}")
        t0 = time.time()
        m = H.measure_bpb(fwd, ids, 2048, 1024, total_bytes, device, batch_size=4,
                          progress_path=OUT + ".gdn.progress")
        m["seconds"] = round(time.time() - t0, 1)
        m["delta_nats_vs_harness"] = m["nats_per_token"] - spec["ref_nats"]
        m["delta_bpb_vs_harness"] = m["bpb"] - spec["ref_bpb"]
        m["match_within_0.01_nats"] = abs(m["delta_nats_vs_harness"]) < 0.01
        r["ymode_full"] = m
        log(f"GDN FULL nats={m['nats_per_token']:.4f} bpb={m['bpb']:.4f} "
            f"d={m['delta_nats_vs_harness']:+.5f} match={m['match_within_0.01_nats']} ({m['seconds']}s)")
        dst = stage_gdn(hf); r["staged_dir"] = dst; log(f"GDN staged -> {dst}")
        del hf, fwd; torch.cuda.empty_cache()
    except Exception:
        r.setdefault("ymode_full", {})["error"] = traceback.format_exc()
        log("GDN RERUN FAILED:\n" + r["ymode_full"]["error"])
    res[name] = r; save_res(res)

def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    shim_ndm()
    device = torch.device("cuda")
    ids, total_bytes = load_ids()
    log(f"slice ok {total_bytes} bytes {ids.size(1)} tokens; GPU {torch.cuda.get_device_name(0)}")
    res = load_res()

    # 1. GDN re-run
    log("===== GDN y-mode full-slice re-run (batch 4, expandable segments) =====")
    rerun_gdn(ids, total_bytes, device, res)

    # 2. packaging patch + safetensors metadata fix on every staged dir
    for name in ALL:
        dst = os.path.join(STAGE_ROOT, name)
        if not os.path.isdir(dst):
            log(f"{name}: no staged dir, skip patch"); continue
        try:
            pst = patch_modeling(dst)
            nt = fix_safetensors_metadata(dst)
            log(f"{name}: modeling {pst}; safetensors format-fixed ({nt} tensors)")
        except Exception:
            log(f"{name}: patch/metadata FAILED:\n" + traceback.format_exc())

    # 3. from_pretrained reload + generation for each
    for name in ALL:
        if not os.path.isdir(os.path.join(STAGE_ROOT, name)):
            continue
        log(f"===== reload check: {name} =====")
        rc = reload_check(name, ids, device)
        res.setdefault(name, {})["reload"] = rc
        save_res(res)
        log(f"{name}: reload={rc.get('from_pretrained')} block={rc.get('reload_block_nats')} "
            f"5win={rc.get('reload_5win_nats')} gen={rc.get('generation')}")
    log("FOLLOWUP DONE")
    save_res(res)

if __name__ == "__main__":
    main()
