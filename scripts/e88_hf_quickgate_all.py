#!/usr/bin/env python3
"""Quick (few-window) sanity gate on the HF v0.3 weights for ALL THREE models via
the genuine HF modeling code (NdmForCausalLM + ndm.models.* forward), finalizer
bypassed. Confirms whether the x-mode/catastrophic blocker applies to each model
rather than assuming. GPU 0 only. -> /tmp/e88_hf_quickgate_all.json
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
import sys, glob, json, math, time, hashlib, importlib, types, traceback
sys.path.insert(0, "/home/erikg/elman"); sys.path.insert(0, "/home/erikg/elman/elman/cuda")
import torch, torch.nn.functional as F
from safetensors.torch import load_file

OUT = "/tmp/e88_hf_quickgate_all.json"
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"
MODELS = [
    {"name": "emender-e88-1.3b", "repo": "poietic-pbc/emender-e88-1.3b", "rev": "v0.3"},
    {"name": "gdn-1.3b",         "repo": "poietic-pbc/gdn-1.3b",         "rev": "v0.3"},
    {"name": "m2rnn-cma-1.3b",   "repo": "poietic-pbc/m2rnn-cma-1.3b",   "rev": "v0.3"},
]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def shim():
    ll=importlib.import_module("elman.models.ladder_lm"); mb=importlib.import_module("elman.models.m2rnn_baseline")
    ndm=types.ModuleType("ndm"); ndmm=types.ModuleType("ndm.models"); ndm.models=ndmm
    sys.modules.update({"ndm":ndm,"ndm.models":ndmm,"ndm.models.ladder_lm":ll,"ndm.models.m2rnn_baseline":mb})

def build(spec):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg=AutoConfig.from_pretrained(spec["repo"],revision=spec["rev"],trust_remote_code=True)
    cls=get_class_from_dynamic_module("modeling_ndm.NdmForCausalLM",spec["repo"],revision=spec["rev"])
    model=cls(cfg)
    snap=glob.glob(f"/home/erikg/.cache/huggingface/hub/models--{spec['repo'].replace('/','--')}/snapshots/*/model.safetensors")[0]
    missing,unexpected=model.load_state_dict(load_file(snap), strict=False)
    log(f"  load: missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(torch.bfloat16).cuda().eval()

@torch.no_grad()
def gate(spec, text, nwin=5):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(spec["repo"],revision=spec["rev"],trust_remote_code=True)
    ids=tok(text[:120000],add_special_tokens=False,return_tensors="pt")["input_ids"][0].cuda()
    model=build(spec)
    total=0.0; n=0; prev=0; w=0
    for begin in range(0, ids.numel(), 1024):
        end=min(begin+2048, ids.numel()); trg=end-prev
        lg=model(ids[begin:end].unsqueeze(0)).logits[0].float()
        tgt=ids[begin:end]; keep=torch.arange(begin+1,end,device=ids.device)>=(end-trg)
        sl=lg[:-1][keep]; st=tgt[1:][keep]
        total+=F.cross_entropy(sl,st,reduction="sum").item(); n+=int(st.numel()); prev=end; w+=1
        if w>=nwin: break
        if end==ids.numel(): break
    del model; torch.cuda.empty_cache()
    return total/max(n,1)

def main():
    shim()
    text=open(SLICE,"rb").read().decode("utf-8")
    res={}
    for spec in MODELS:
        try:
            log(f"=== {spec['name']} HF-weights quick gate ===")
            nats=gate(spec, text)
            res[spec["name"]]={"nats_per_token_quick": nats,
                               "verdict": "SANE" if nats<5 else "CATASTROPHIC_xmode"}
            log(f"  {spec['name']}: {nats:.4f} nats/tok -> {res[spec['name']]['verdict']}")
        except Exception:
            res[spec["name"]]={"error": traceback.format_exc()}
            log("  FAILED:\n"+traceback.format_exc())
        json.dump(res, open(OUT,"w"), indent=2)
    log("QUICKGATE_ALL DONE "+json.dumps({k:v.get('nats_per_token_quick',v.get('error','?')) if isinstance(v,dict) else v for k,v in res.items()}))

if __name__=="__main__":
    main()
