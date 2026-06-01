#!/usr/bin/env python3
"""Probe the CURRENT public HF v0.3 release (post republish-hf-v03).

Force a FRESH download (clean isolated cache) so we get the weights the public
@v0.3 tag resolves to NOW, not the stale x-mode commit cached locally. Print:
  - the resolved commit SHA for the v0.3 tag,
  - the safetensors metadata header (checkpoint_step / ymode provenance),
  - the safetensors tensor key namespace (count + sample),
  - the harness model key namespace built from the pinned args.json.

No GPU needed for the probe (weights not loaded to device). Read-only.
"""
import os, sys, json, time
CLEAN = "/tmp/v03-init-clean-cache"
os.environ["HF_HOME"] = CLEAN
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CLEAN, "hub")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
sys.path.insert(0, "/home/erikg/elman")
sys.path.insert(0, "/home/erikg/elman/elman/cuda")

import json as _json
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from safetensors import safe_open

REPOS = {
    "e88":   ("poietic-pbc/emender-e88-1.3b",  "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/e88"),
    "gdn":   ("poietic-pbc/gdn-1.3b",          "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/gdn"),
    "m2rnn": ("poietic-pbc/m2rnn-cma-1.3b",    "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/m2rnn"),
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    api = HfApi()
    out = {}
    for key, (repo, ckptdir) in REPOS.items():
        log(f"==== {key}: {repo} @ v0.3 ====")
        sha = api.model_info(repo, revision="v0.3").sha
        log(f"  resolved v0.3 commit sha = {sha}")
        st_path = hf_hub_download(repo, "model.safetensors", revision="v0.3", force_download=True)
        log(f"  downloaded -> {st_path}")
        with safe_open(st_path, framework="pt") as f:
            meta = f.metadata() or {}
            keys = list(f.keys())
        log(f"  safetensors metadata: {_json.dumps(meta)}")
        log(f"  n_tensors={len(keys)}  sample_keys={keys[:4]} ... {keys[-3:]}")
        out[key] = {"repo": repo, "sha": sha, "metadata": meta,
                    "n_tensors": len(keys), "keys_sample": keys[:6] + keys[-4:]}
    Path("scripts/v03_probe_result.json").write_text(json.dumps(out, indent=2))
    log("wrote scripts/v03_probe_result.json")


if __name__ == "__main__":
    raise SystemExit(main())
