#!/usr/bin/env python3
"""Full canonical-slice load-sanity for the v0.3-initialised harness model.

Strict-loads the public @v0.3 safetensors into the trainable elman-harness model
and scores the FULL canonical held-out slice (ctx 2048 / stride 1024) -> nats and
BPB. Since v03_init_verify proved the v0.3 init is bit-identical to the pinned
y-mode init, this should reproduce the published readback (E88 2.559775 / GDN
2.559748 / M2RNN 2.547022 nats) exactly. Self-contained confirmation. GPU pinned.
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
CLEAN = "/tmp/v03-init-clean-cache"
os.environ["HF_HOME"] = CLEAN
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CLEAN, "hub")
sys.path.insert(0, "/home/erikg/elman")
sys.path.insert(0, "/home/erikg/elman/elman/cuda")
sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-757/scripts")
sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-777/scripts")

from pathlib import Path
import torch
import measure_pile_bpb_elman as H
import hf_v03_republish_lib as L
from v03_init_verify import build_model, load_v03_into_harness
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json as _json

REF = {"e88": 2.5597944649733653, "gdn": 2.5597479088882387, "m2rnn": 2.547022302196617}
DIRS = {"e88": "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/e88",
        "gdn": "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/gdn",
        "m2rnn": "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/m2rnn"}
REPOS = {"e88": "poietic-pbc/emender-e88-1.3b", "gdn": "poietic-pbc/gdn-1.3b",
         "m2rnn": "poietic-pbc/m2rnn-cma-1.3b"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    import tiktoken
    device = torch.device("cuda")
    ids, total_bytes = L.load_ids()
    log(f"slice {total_bytes} bytes {ids.size(1)} tokens device={torch.cuda.get_device_name(0)}")
    out = {}
    for key in ["e88", "gdn", "m2rnn"]:
        a_json = json.loads(Path(DIRS[key], "args.json").read_text())
        vocab = tiktoken.get_encoding(a_json["tokenizer"]).n_vocab
        m = build_model(a_json, vocab)
        st_path = hf_hub_download(REPOS[key], "model.safetensors", revision="v0.3")
        load_v03_into_harness(m, load_file(st_path))
        m = m.to(device).bfloat16().eval()
        block = float(H.block_loss_sanity(m, ids, 2048, device))
        res = H.measure_bpb(m, ids, 2048, 1024, total_bytes, device, batch_size=8)
        res["block_nats"] = block
        res["ref_nats"] = REF[key]
        res["delta_vs_ref"] = res["nats_per_token"] - REF[key]
        res["gate_pass"] = bool(2.4 <= res["nats_per_token"] <= 2.7 and abs(res["delta_vs_ref"]) < 0.01)
        out[key] = res
        log(f"{key}: full-slice nats={res['nats_per_token']:.6f} bpb={res['bpb']:.6f} "
            f"d_vs_ref={res['delta_vs_ref']:+.2e} gate={res['gate_pass']}")
        del m; torch.cuda.empty_cache()
        Path("scripts/v03_loadsanity_fullslice_result.json").write_text(_json.dumps(out, indent=2))
    log("wrote scripts/v03_loadsanity_fullslice_result.json")


if __name__ == "__main__":
    raise SystemExit(main())
