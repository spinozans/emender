#!/usr/bin/env python3
"""republish-hf-v03 step 3 — POST-UPLOAD readback from the PUBLIC @v0.3.

The whole point: prove the public release now WORKS. From a CLEAN cache,
AutoModelForCausalLM.from_pretrained(repo, revision='v0.3', trust_remote_code=
True) for each repo, run the FULL canonical slice (ctx 2048 / stride 1024,
9,999,511-byte denominator) through the genuine bundled NdmForCausalLM.forward,
and confirm sane (~2.56 nats, ~0.966 bpb, gate PASS). GPU 0 only.

A fresh, isolated HF cache dir is used so nothing is served from a stale local
snapshot — the bytes are pulled from the public hub at the v0.3 tag as it
resolves AFTER the overwrite.

The only environment shim is mapping `ndm.models.*` -> elman (the documented
local-verification shim; not a repo change). transformers==4.57.x needs no
tie_weights patch (the published tie_weights(self) + _tied_weights_keys work).

Result JSON: scripts/hf_v03_republish_readback_result.json
"""
import os, sys, json, time, traceback, tempfile
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# force a clean, isolated cache so the readback truly re-downloads @v0.3
CLEAN_CACHE = os.environ.get("REPUBLISH_CLEAN_CACHE",
                             "/tmp/republish-v03-readback-clean-cache")
os.environ["HF_HOME"] = CLEAN_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CLEAN_CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CLEAN_CACHE, "hub")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hf_v03_republish_lib as L
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_v03_republish_readback_result.json")


def main():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert cvd and "," not in cvd, f"pin a single GPU via CUDA_VISIBLE_DEVICES (got {cvd!r})"
    os.makedirs(os.path.join(CLEAN_CACHE, "hub"), exist_ok=True)
    L.shim_ndm()
    from transformers import AutoModelForCausalLM
    import measure_pile_bpb_elman as H
    device = torch.device("cuda")
    ids, total_bytes = L.load_ids()
    L.log(f"CLEAN cache={CLEAN_CACHE}; slice {total_bytes} bytes {ids.size(1)} tokens; "
          f"device={torch.cuda.get_device_name(0)}")
    # resumable: keep prior sane results so a GPU-contention OOM retry only
    # re-runs the models that have not yet passed
    res = {"_meta": {"clean_cache": CLEAN_CACHE, "slice_bytes": total_bytes,
                     "tokens": ids.size(1), "context": 2048, "stride": 1024,
                     "path": "AutoModelForCausalLM.from_pretrained(repo, revision='v0.3', trust_remote_code=True)"}}
    if os.path.exists(OUT):
        try:
            prior = json.load(open(OUT))
            for k, v in prior.items():
                if isinstance(v, dict) and v.get("gate_pass"):
                    res[k] = v
        except Exception:
            pass
    json.dump(res, open(OUT, "w"), indent=2)
    all_sane = True
    for spec in L.MODELS:
        name, repo = spec["name"], spec["repo"]
        if isinstance(res.get(name), dict) and res[name].get("gate_pass"):
            L.log(f"==== {repo}@v0.3 already SANE (skipping) ====")
            continue
        L.log(f"==== {repo}@v0.3 (public readback) ====")
        r = {"repo": repo, "ref_nats": spec["ref_nats"], "ref_bpb": spec["ref_bpb"]}
        try:
            t0 = time.time()
            m = AutoModelForCausalLM.from_pretrained(repo, revision="v0.3", trust_remote_code=True,
                                                     dtype=torch.bfloat16)
            m = m.to(device).eval()
            r["load_seconds"] = round(time.time() - t0, 1)
            fwd = L.HFForward(m)
            r["block_nats"] = float(H.block_loss_sanity(fwd, ids, 2048, device))
            t1 = time.time()
            mres = H.measure_bpb(fwd, ids, 2048, 1024, total_bytes, device, batch_size=8,
                                 progress_path=OUT + f".{name}.progress")
            mres["seconds"] = round(time.time() - t1, 1)
            mres["delta_nats_vs_harness"] = mres["nats_per_token"] - spec["ref_nats"]
            mres["delta_bpb_vs_harness"] = mres["bpb"] - spec["ref_bpb"]
            gate = bool(2.4 <= mres["nats_per_token"] <= 2.7 and abs(mres["delta_nats_vs_harness"]) < 0.01)
            r["full_slice"] = mres
            r["gate_pass"] = gate
            r["verdict"] = "SANE" if gate else "BROKEN"
            all_sane = all_sane and gate
            L.log(f"{name} READBACK: block={r['block_nats']:.4f} nats={mres['nats_per_token']:.6f} "
                  f"bpb={mres['bpb']:.6f} d_nats={mres['delta_nats_vs_harness']:+.2e} "
                  f"gate={gate} ({mres['seconds']}s)")
            del m, fwd; torch.cuda.empty_cache()
        except Exception:
            r["error"] = traceback.format_exc()
            all_sane = False
            L.log(f"{name} READBACK FAILED:\n{r['error']}")
        res[name] = r
        json.dump(res, open(OUT, "w"), indent=2)
    all_sane = all(isinstance(res.get(s["name"]), dict) and res[s["name"]].get("gate_pass")
                   for s in L.MODELS)
    res["_all_sane"] = all_sane
    json.dump(res, open(OUT, "w"), indent=2)
    L.log(f"ALL READBACK SANE = {all_sane}")
    return 0 if all_sane else 1


if __name__ == "__main__":
    raise SystemExit(main())
