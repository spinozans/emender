#!/usr/bin/env python3
"""Discriminate the root cause of the HF v0.3 E88 forward giving 19.6 nats/token.

Hypotheses:
  H_weights : the HF safetensors hold schedule-free x-mode weights (catastrophic
              at inference; the sibling task showed standalone model_state_dict ->
              ~17.6 nats). HF artifact has no optimizer state, so no y-mode swap.
  H_config  : the HF config ships r_h_mode="auto" (the HF wrapper passes it raw to
              LadderLM), whereas train.py RESOLVES r_h_mode before constructing
              (E88 -> "none"). A wrong r_h_mode could change compute.

Test: build E88 with the SAME live-harness code the sibling validated as
known-good (elman LadderLM, r_h_mode RESOLVED like train.py), load the HF v0.3
safetensors weights into it (NOT the local .pt), and measure a few windows.
  * still ~19.6 nats  -> H_weights (HF weights are x-mode; config is not the cause)
  * ~2.6 nats         -> H_config (HF r_h_mode="auto" is the bug; weights are fine)

GPU 0 ONLY. Result -> /tmp/e88_rootcause_result.json
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
import sys, glob, json, math, traceback, time
sys.path.insert(0, "/home/erikg/elman")
sys.path.insert(0, "/home/erikg/elman/elman/cuda")
import torch, torch.nn.functional as F
from safetensors.torch import load_file

OUT = "/tmp/e88_rootcause_result.json"
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"
SAFET = glob.glob("/home/erikg/.cache/huggingface/hub/models--poietic-pbc--emender-e88-1.3b/snapshots/*/model.safetensors")[0]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def build(r_h_mode, use_triton):
    from elman.models import LadderLM
    return LadderLM(vocab_size=50281, dim=1664, depth=12, level="E88", expansion=1.0,
        n_groups=32, n_state=32, n_slots=64, n_heads=370, top_k=None, k_fast=None, k_slow=None,
        use_gate=True, gate_activation="silu", linear_state=False, use_write_gate=False,
        e88_decay_mode="mamba", e88_value_residual=False, state_expansion=2, r_h_mode=r_h_mode,
        use_conv=False, d_conv=4, dropout=0.0, checkpoint_interval=16,
        gradient_checkpointing=False, projection_chunk_size=0, loss_chunk_size=0,
        use_triton=use_triton)

@torch.no_grad()
def quick_nats(model, ids, nwin=3):
    total=0.0; n=0; prev=0; w=0
    for begin in range(0, ids.numel(), 1024):
        end=min(begin+2048, ids.numel()); trg=end-prev
        lg=model(ids[begin:end].unsqueeze(0)).logits[0].float() if hasattr(model(ids[begin:end].unsqueeze(0)),'logits') else None
        prev=end; w+=1
        if w>=nwin: break
    return None  # placeholder, replaced below

def measure(model, ids, nwin=3):
    total=0.0; n=0; prev=0; w=0; t0=time.time()
    with torch.no_grad():
        for begin in range(0, ids.numel(), 1024):
            end=min(begin+2048, ids.numel()); trg=end-prev
            out=model(ids[begin:end].unsqueeze(0))
            logits=(out.logits if hasattr(out,"logits") else out)[0].float()
            tgt=ids[begin:end]
            keep=torch.arange(begin+1,end,device=ids.device) >= (end-trg)
            sl=logits[:-1][keep]; st=tgt[1:][keep]
            total+=F.cross_entropy(sl,st,reduction="sum").item(); n+=int(st.numel())
            prev=end; w+=1
            log(f"    win{w} nats/tok={total/max(n,1):.4f} ({time.time()-t0:.0f}s)")
            if w>=nwin: break
            if end==ids.numel(): break
    return total/max(n,1)

def load_hf_into(model):
    sd=load_file(SAFET)
    sd2={k[len("model."):] if k.startswith("model.") else k: v for k,v in sd.items()}
    missing,unexpected=model.load_state_dict(sd2, strict=False)
    log(f"  load HF weights: missing={len(missing)} unexpected={len(unexpected)} "
        f"miss[:3]={missing[:3]} unexp[:3]={unexpected[:3]}")
    return model

def main():
    res={}
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained("poietic-pbc/emender-e88-1.3b",revision="v0.3",trust_remote_code=True)
        text=open(SLICE,"rb").read().decode("utf-8")
        ids=tok(text[:80000],add_special_tokens=False,return_tensors="pt")["input_ids"][0][:6144].cuda()
        log(f"ids {ids.numel()}")
        for rhm in ["none","auto"]:
            try:
                log(f"=== build r_h_mode={rhm} use_triton=True + HF weights ===")
                m=build(rhm, True)
                m=load_hf_into(m).to(torch.bfloat16).cuda().eval()
                nats=measure(m, ids, nwin=3)
                res[f"r_h_mode={rhm}"]=nats
                log(f"  r_h_mode={rhm}: nats/tok={nats:.4f}")
                del m; torch.cuda.empty_cache()
            except Exception:
                res[f"r_h_mode={rhm}_error"]=traceback.format_exc()
                log("  FAILED:\n"+traceback.format_exc())
            json.dump(res, open(OUT,"w"), indent=2)
    except Exception:
        res["error"]=traceback.format_exc(); log(res["error"])
    json.dump(res, open(OUT,"w"), indent=2)
    log("ROOTCAUSE DONE "+json.dumps({k:v for k,v in res.items() if not k.endswith('error')}))

if __name__=="__main__":
    main()
