#!/usr/bin/env python3
"""Decisive test: is the HF v0.3 export corrupt, or are even the source .pt weights
x-mode? Compare the HF safetensors against a source training .pt model_state_dict,
and measure BOTH through the SAME live-harness LadderLM forward (quick gate).

  source .pt sane (~2.6) AND HF != source  -> the HF EXPORT corrupted the weights;
                                              re-export from source .pt is the fix.
  source .pt also ~17 nats                  -> source weights are x-mode too; the
                                              fix needs the y-mode optimizer swap
                                              (which these .pt files do NOT contain).

GPU 0 only. -> /tmp/e88_export_vs_source.json
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
import sys, glob, json, time, traceback
sys.path.insert(0, "/home/erikg/elman"); sys.path.insert(0, "/home/erikg/elman/elman/cuda")
import torch, torch.nn.functional as F
from safetensors.torch import load_file

OUT = "/tmp/e88_export_vs_source.json"
SLICE = "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt"
HF_ST = glob.glob("/home/erikg/.cache/huggingface/hub/models--poietic-pbc--emender-e88-1.3b/snapshots/*/model.safetensors")[0]
SRC_PT = "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1539000_loss_2.6070.pt"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def build():
    from elman.models import LadderLM
    return LadderLM(vocab_size=50281, dim=1664, depth=12, level="E88", expansion=1.0,
        n_groups=32, n_state=32, n_slots=64, n_heads=370, top_k=None, k_fast=None, k_slow=None,
        use_gate=True, gate_activation="silu", linear_state=False, use_write_gate=False,
        e88_decay_mode="mamba", e88_value_residual=False, state_expansion=2, r_h_mode="none",
        use_conv=False, d_conv=4, dropout=0.0, checkpoint_interval=16,
        gradient_checkpointing=False, projection_chunk_size=0, loss_chunk_size=0, use_triton=True)

def strip_model_prefix(sd):
    return {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}

@torch.no_grad()
def quick_nats(model, ids, nwin=4):
    total=0.0; n=0; prev=0; w=0
    for begin in range(0, ids.numel(), 1024):
        end=min(begin+2048, ids.numel()); trg=end-prev
        out=model(ids[begin:end].unsqueeze(0))
        lg=(out.logits if hasattr(out,"logits") else out)[0].float()
        tgt=ids[begin:end]; keep=torch.arange(begin+1,end,device=ids.device)>=(end-trg)
        sl=lg[:-1][keep]; st=tgt[1:][keep]
        total+=F.cross_entropy(sl,st,reduction="sum").item(); n+=int(st.numel()); prev=end; w+=1
        if w>=nwin: break
        if end==ids.numel(): break
    return total/max(n,1)

def main():
    res={}
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained("poietic-pbc/emender-e88-1.3b",revision="v0.3",trust_remote_code=True)
        ids=tok(open(SLICE,"rb").read().decode("utf-8")[:90000],add_special_tokens=False,return_tensors="pt")["input_ids"][0].cuda()
        log(f"ids {ids.numel()}")

        hf=strip_model_prefix(load_file(HF_ST))
        src_raw=torch.load(SRC_PT, map_location="cpu", weights_only=False)["model_state_dict"]
        src=strip_model_prefix(src_raw)
        log(f"HF keys={len(hf)} SRC keys={len(src)}")

        # compare overlapping tensors
        common=[k for k in hf if k in src and src[k].shape==hf[k].shape]
        diffs={}
        nz=0
        for k in common:
            d=(hf[k].float()-src[k].float()).abs()
            md=d.max().item(); mn=d.mean().item()
            if md>1e-6: nz+=1
            if md>1e-3:
                diffs[k]={"max_abs_diff":md,"mean_abs_diff":mn}
        res["overlap_keys"]=len(common)
        res["keys_with_any_diff(>1e-6)"]=nz
        res["keys_with_big_diff(>1e-3)"]=len(diffs)
        res["sample_big_diffs"]={k:diffs[k] for k in list(diffs)[:6]}
        log(f"overlap={len(common)} any_diff={nz} big_diff={len(diffs)}")
        json.dump(res, open(OUT,"w"), indent=2)

        # measure source .pt weights through live forward
        m=build()
        miss,unexp=m.load_state_dict(src, strict=False)
        log(f"src load: missing={len(miss)} unexpected={len(unexp)}")
        m=m.to(torch.bfloat16).cuda().eval()
        res["source_pt_nats"]=quick_nats(m, ids)
        res["source_pt_verdict"]="SANE" if res["source_pt_nats"]<5 else "xmode/broken"
        log(f"SOURCE .pt nats/tok={res['source_pt_nats']:.4f} -> {res['source_pt_verdict']}")
        del m; torch.cuda.empty_cache()
        json.dump(res, open(OUT,"w"), indent=2)

        # measure HF safetensors weights through the SAME live forward (control)
        m=build()
        miss,unexp=m.load_state_dict(hf, strict=False)
        log(f"hf load: missing={len(miss)} unexpected={len(unexp)}")
        m=m.to(torch.bfloat16).cuda().eval()
        res["hf_safetensors_nats"]=quick_nats(m, ids)
        res["hf_safetensors_verdict"]="SANE" if res["hf_safetensors_nats"]<5 else "xmode/broken"
        log(f"HF safetensors nats/tok={res['hf_safetensors_nats']:.4f} -> {res['hf_safetensors_verdict']}")
        del m; torch.cuda.empty_cache()
    except Exception:
        res["error"]=traceback.format_exc(); log(res["error"])
    json.dump(res, open(OUT,"w"), indent=2)
    log("EXPORT_VS_SOURCE DONE")

if __name__=="__main__":
    main()
