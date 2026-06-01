import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0"); os.environ.setdefault("XMA_PATH","/home/erikg/xma")
import sys, json, types, importlib, traceback
sys.path.insert(0,"/home/erikg/elman"); sys.path.insert(0,"/home/erikg/elman/elman/cuda")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ll=importlib.import_module("elman.models.ladder_lm"); mb=importlib.import_module("elman.models.m2rnn_baseline")
ndm=types.ModuleType("ndm"); ndmm=types.ModuleType("ndm.models"); ndm.models=ndmm
sys.modules.update({"ndm":ndm,"ndm.models":ndmm,"ndm.models.ladder_lm":ll,"ndm.models.m2rnn_baseline":mb})
OUT="/home/erikg/ndm/.wg-worktrees/agent-757/scripts/hf_v03_fix_verify_result.json"
STAGE="/home/erikg/ndm/.wg-worktrees/agent-757/hf_v03_fix_staging"
res=json.load(open(OUT))
dev=torch.device("cuda")
for name in ["emender-e88-1.3b","gdn-1.3b","m2rnn-cma-1.3b"]:
    dst=os.path.join(STAGE,name)
    try:
        m=AutoModelForCausalLM.from_pretrained(dst,trust_remote_code=True,dtype=torch.bfloat16).to(dev).eval()
        tok=AutoTokenizer.from_pretrained(dst,trust_remote_code=True)
        ids=tok("The history of science", return_tensors="pt").input_ids.to(dev)
        g=m.generate(ids, max_new_tokens=24, do_sample=False, use_cache=False)
        txt=tok.decode(g[0])
        ok=g.shape[1]==ids.shape[1]+24
        res.setdefault(name,{}).setdefault("reload",{})["generation"]={"ok":bool(ok),"out_len":int(g.shape[1]),"sample":txt[:160]}
        print(f"{name}: gen ok={ok} len={g.shape[1]} :: {txt[:120]!r}",flush=True)
        del m; torch.cuda.empty_cache()
    except Exception:
        res.setdefault(name,{}).setdefault("reload",{})["generation"]={"ok":False,"error":traceback.format_exc().splitlines()[-1]}
        print(f"{name}: gen FAILED\n"+traceback.format_exc(),flush=True)
    json.dump(res,open(OUT,"w"),indent=2)
print("GENTEST DONE")
