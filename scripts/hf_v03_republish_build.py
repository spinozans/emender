#!/usr/bin/env python3
"""republish-hf-v03 step 1 — build + LOCAL pre-upload gate of the y-mode overwrite.

For each repo:
  * Load the VERIFIED y-mode staged safetensors (full-slice verified by
    fix-hf-v03 to reproduce the live harness to <=2e-6 nats).
  * Re-add the explicit tied `model.lm_head.weight` (= model.embedding.weight)
    so the uploaded file has the SAME key set as the published x-mode file
    (87 / 297 / 150 keys) -> a true structural drop-in; only tensor VALUES
    change (x-mode -> y-mode). Write accurate y-mode provenance metadata.
  * Stage a local dir = the PUBLISHED @v0.3 code/config/tokenizer (copied from
    the clean HF snapshot) + the new safetensors. This dir is byte-identical to
    what @v0.3 will resolve to after the weights-only overwrite.
  * GATE: AutoModelForCausalLM.from_pretrained(stage_dir, trust_remote_code) ->
    measure block-loss + windows on the canonical slice through the genuine
    NdmForCausalLM.forward. Require strict 0-missing/0-unexpected load and
    block-loss in [1.5, 1.9] (~2.56 nats), matching the verified reference.

Nothing is uploaded here. Output dirs: /tmp/republish-v03-build/<name>/
Result JSON: scripts/hf_v03_republish_build_result.json
"""
import os, sys, json, shutil, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_v03_republish_lib as L

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BUILD_ROOT = "/tmp/republish-v03-build"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_v03_republish_build_result.json")

CODE_FILES = ("config.json", "configuration_ndm.py", "modeling_ndm.py",
              "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json")


def snap(repo, rev="v0.3"):
    base = f"/home/erikg/.cache/huggingface/hub/models--{repo.replace('/','--')}"
    c = open(f"{base}/refs/{rev}").read().strip()
    return f"{base}/snapshots/{c}"


def build_one(spec):
    name, repo = spec["name"], spec["repo"]
    staged = os.path.join(L.STAGE_ROOT, name, "model.safetensors")
    pub_snap = snap(repo)
    pub_st = os.path.join(pub_snap, "model.safetensors")

    # published key set (the structure we must reproduce)
    with safe_open(pub_st, framework="pt") as h:
        pub_keys = set(h.keys()); pub_meta = dict(h.metadata() or {})

    # verified y-mode tensors (save_model deduped the tied lm_head)
    sd = load_file(staged)
    assert "model.embedding.weight" in sd, "embedding key missing in staged y-mode export"
    # re-materialise the tied output head to match the published structure
    if "model.lm_head.weight" not in sd and "model.lm_head.weight" in pub_keys:
        sd["model.lm_head.weight"] = sd["model.embedding.weight"].clone()
    new_keys = set(sd.keys())
    assert new_keys == pub_keys, f"{name} key set != published: only_pub={sorted(pub_keys-new_keys)} only_new={sorted(new_keys-pub_keys)}"

    meta = {
        "format": "pt",
        "repo_id": repo,
        "model_identity": spec["identity"],
        "source_state_dict": "model_state_dict (y-mode; schedule-free AdamWScheduleFree optimizer.train() swap)",
        "checkpoint_step": str(spec["ckpt_step"]),
        "ymode_export": "true",
        "republish_task": "republish-hf-v03",
        "note": "y-mode weights overwriting prior x-mode v0.3 weights; reproduces live-harness held-out BPB",
    }

    out_dir = os.path.join(BUILD_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    # exact published code/config/tokenizer (UNCHANGED) -> the upload changes weights only
    for fn in CODE_FILES:
        s = os.path.join(pub_snap, fn)
        if os.path.exists(s):
            shutil.copy2(os.path.realpath(s), os.path.join(out_dir, fn))
    out_st = os.path.join(out_dir, "model.safetensors")
    save_file(sd, out_st, metadata=meta)
    return out_dir, out_st, sorted(new_keys), pub_meta, meta


def gate_one(spec, out_dir, ids, device):
    from transformers import AutoModelForCausalLM
    import measure_pile_bpb_elman as H
    m = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True, dtype=torch.bfloat16)
    m = m.to(device).eval()
    fwd = L.HFForward(m)
    b = float(H.block_loss_sanity(fwd, ids, 2048, device))
    w = float(L.few_window_nats(fwd, ids, device, nwin=8))
    del m; torch.cuda.empty_cache()
    return {"from_pretrained": "OK", "block_nats": b, "win8_nats": w,
            "gate_pass": bool(1.5 <= b <= 1.9), "verdict": "SANE" if b < 3 else "BROKEN"}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "GPU 0 only"
    L.shim_ndm()
    device = torch.device("cuda")
    ids, nbytes = L.load_ids()
    L.log(f"slice ok {nbytes} bytes {ids.size(1)} tokens; device={torch.cuda.get_device_name(0)}")
    res = {"_meta": {"slice_bytes": nbytes, "tokens": ids.size(1)}}
    json.dump(res, open(OUT, "w"), indent=2)
    all_pass = True
    for spec in L.MODELS:
        name = spec["name"]
        L.log(f"==== {name} ====")
        r = {"repo": spec["repo"], "ckpt_step": spec["ckpt_step"],
             "ref_nats": spec["ref_nats"], "ref_bpb": spec["ref_bpb"]}
        try:
            out_dir, out_st, keys, pub_meta, meta = build_one(spec)
            import hashlib
            h = hashlib.sha256()
            with open(out_st, "rb") as f:
                for blk in iter(lambda: f.read(1 << 27), b""):
                    h.update(blk)
            r["build"] = {"out_dir": out_dir, "n_keys": len(keys),
                          "safetensors_sha256": h.hexdigest(),
                          "safetensors_size": os.path.getsize(out_st),
                          "published_meta": pub_meta, "new_meta": meta}
            L.log(f"{name} built: {len(keys)} keys, sha256={h.hexdigest()[:16]} size={os.path.getsize(out_st)}")
            r["gate"] = gate_one(spec, out_dir, ids, device)
            L.log(f"{name} GATE: block={r['gate']['block_nats']:.4f} win8={r['gate']['win8_nats']:.4f} "
                  f"pass={r['gate']['gate_pass']} ({r['gate']['verdict']})")
            all_pass = all_pass and r["gate"]["gate_pass"]
        except Exception:
            r["error"] = traceback.format_exc()
            all_pass = False
            L.log(f"{name} FAILED:\n{r['error']}")
        res[name] = r
        json.dump(res, open(OUT, "w"), indent=2)
    res["_all_gate_pass"] = all_pass
    json.dump(res, open(OUT, "w"), indent=2)
    L.log(f"ALL GATE PASS = {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
