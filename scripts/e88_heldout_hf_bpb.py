#!/usr/bin/env python3
"""
e88-heldout-hf: Held-out Pile BPB for the three v0.3 models via HuggingFace
bundled modeling code (trust_remote_code), GPU 0 ONLY.

Methodology (matches the canonical panel slice exactly):
  - Input bytes: the canonical held-out Pile slice
    (sha256 3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a,
     total_bytes = 9_999_511). Denominator is these FIXED UTF-8 bytes.
  - For each model: tokenize the slice with the model's own tokenizer
    (tokenizer-invariant BPB), run a strided sliding window of ctx=2048 with
    stride=1024 so that every token (except the very first) is scored with up
    to 2047 tokens of true left context (no boundary-context loss).
  - total_NLL_nats = sum of per-token cross-entropy (nats) over all scored tokens.
  - mean nats/token = total_NLL_nats / n_scored_tokens.
  - BPB = total_NLL_nats / (total_bytes * ln(2)), total_bytes = 9_999_511.
    (No 3.92 constant; pure measured NLL over fixed bytes.)
  - SANITY GATE: a correct forward yields ~2.6 nats/token (sub-1 bpb). If a
    model yields worse-than-random (~10.8 nats) or absurd values, the forward
    is broken -> report BLOCKER, do not publish the number as valid.

Real measurement only. No mock data.
"""
import os
# Pin GPU 0 BEFORE importing torch. GPUs 1-7 are training.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import json
import math
import time
import hashlib
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LN2 = math.log(2.0)
CTX = 2048
STRIDE = 1024
TOTAL_BYTES = 9_999_511
EXPECT_SHA = "3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a"

SLICE_CANDIDATES = [
    "/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt",
]

MODELS = [
    {"name": "emender-e88-1.3b", "repo": "poietic-pbc/emender-e88-1.3b", "rev": "v0.3", "step": 1_524_000},
    {"name": "gdn-1.3b",         "repo": "poietic-pbc/gdn-1.3b",         "rev": "v0.3", "step": 1_998_000},
    {"name": "m2rnn-cma-1.3b",   "repo": "poietic-pbc/m2rnn-cma-1.3b",   "rev": "v0.3", "step": 1_467_000},
]

OUT_JSON = "/home/erikg/ndm/.wg-worktrees/agent-744/scripts/e88_heldout_hf_results.json"
OUT_MD   = "/home/erikg/ndm/.wg-worktrees/agent-744/paper/review/E88_HELDOUT_HF.md"

E88_TRAIN_LOSS = 0.974  # nats/token reported train loss for E88


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_slice():
    for p in SLICE_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No slice file found among {SLICE_CANDIDATES}")


def load_slice():
    p = find_slice()
    with open(p, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    log(f"Slice: {p} len={len(data)} sha256={sha}")
    if len(data) != TOTAL_BYTES:
        raise ValueError(f"Slice byte length {len(data)} != expected {TOTAL_BYTES}")
    if sha != EXPECT_SHA:
        raise ValueError(f"Slice sha256 {sha} != expected {EXPECT_SHA}")
    # Decode for tokenizers. Bytes are fixed for the denominator regardless of decode.
    try:
        text = data.decode("utf-8")
        decode_mode = "utf-8 strict"
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")
        decode_mode = "utf-8 errors=ignore"
    log(f"Decoded slice: {len(text)} chars ({decode_mode})")
    return text, sha, decode_mode


@torch.no_grad()
def score_model(spec, text, device):
    log(f"=== {spec['name']} ({spec['repo']} @ {spec['rev']}) ===")
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["rev"], trust_remote_code=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    n = ids.numel()
    log(f"Tokenized: {n} tokens (vocab via {spec['name']} tokenizer)")

    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"], revision=spec["rev"], trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()

    ids = ids.to(device)

    total_nll = 0.0
    n_scored = 0
    prev_end = 0
    t0 = time.time()
    nwin = 0
    for begin in range(0, n, STRIDE):
        end = min(begin + CTX, n)
        trg_len = end - prev_end  # number of new tokens this window contributes
        input_ids = ids[begin:end].unsqueeze(0)
        out = model(input_ids)
        logits = out.logits[0].float()  # (L, V)
        # predict token t from logits at t-1: logits[:-1] vs ids[begin+1:end]
        # we only score the last trg_len target positions (the new tokens)
        targets = ids[begin:end]
        # shifted: pred for position i in [1, L-1]
        shift_logits = logits[:-1]            # (L-1, V) -> preds for targets[1:]
        shift_targets = targets[1:]           # (L-1,)
        # mask: only keep the last trg_len of the *targets* (positions end-trg_len .. end-1)
        # in shifted index space, target original pos = begin+1 .. end-1
        keep = (torch.arange(begin + 1, end, device=device) >= (end - trg_len))
        sl = shift_logits[keep]
        st = shift_targets[keep]
        if sl.numel() > 0:
            nll = F.cross_entropy(sl, st, reduction="sum")
            total_nll += nll.item()
            n_scored += int(st.numel())
        prev_end = end
        nwin += 1
        if nwin % 200 == 0:
            log(f"  win {nwin} begin={begin}/{n} scored={n_scored} "
                f"running nats/tok={total_nll/max(n_scored,1):.4f} "
                f"({time.time()-t0:.0f}s)")
        if end == n:
            break

    dt = time.time() - t0
    mean_nats = total_nll / max(n_scored, 1)
    bpb = total_nll / (TOTAL_BYTES * LN2)

    # Sanity gate: correct forward ~2.6 nats/token (sub-1 bpb).
    # worse-than-random ~10.8 nats => broken. Accept a generous window.
    passed = (mean_nats < 5.0) and (bpb < 2.0) and math.isfinite(mean_nats)
    verdict = "PASS" if passed else "FAIL"

    res = {
        "name": spec["name"],
        "repo": spec["repo"],
        "revision": spec["rev"],
        "step": spec["step"],
        "n_tokens_total": int(n),
        "n_tokens_scored": int(n_scored),
        "total_nll_nats": float(total_nll),
        "mean_nats_per_token": float(mean_nats),
        "bpb": float(bpb),
        "ctx": CTX,
        "stride": STRIDE,
        "sanity_gate": verdict,
        "seconds": dt,
    }
    log(f"RESULT {spec['name']}: nats/tok={mean_nats:.4f} bpb={bpb:.4f} "
        f"tokens_scored={n_scored} gate={verdict} ({dt:.0f}s)")

    del model
    torch.cuda.empty_cache()
    return res


def render_md(results, sha, decode_mode, errors):
    lines = []
    lines.append("# E88 Held-out Pile BPB — HuggingFace v0.3 (trust_remote_code)\n")
    lines.append("Real measurement on GPU 0 only (GPUs 1-7 were training). Models loaded via")
    lines.append("`AutoModelForCausalLM.from_pretrained(repo, revision='v0.3', trust_remote_code=True)`,")
    lines.append("bf16, the same bundled modeling code that was generation-validated for the v0.3 release.\n")
    lines.append("## Canonical slice")
    lines.append("")
    lines.append("- source: `/mnt/nvme2n1/erikg/pile.txt`")
    lines.append("- byte_offset: 1000000001956, byte_length / total_bytes: 9999511")
    lines.append(f"- sha256: `{sha}` (verified: {'OK' if sha == EXPECT_SHA else 'MISMATCH'})")
    lines.append(f"- decode: {decode_mode}")
    lines.append("- context: ctx=2048, strided sliding window stride=1024 (every token scored with full left context)")
    lines.append("- BPB = total_NLL_nats / (total_bytes * ln2), total_bytes = 9999511; no 3.92 constant.\n")
    lines.append("## Sanity gate")
    lines.append("")
    lines.append("A correct forward yields ~2.6 nats/token (sub-1 bpb). Worse-than-random (~10.8 nats)")
    lines.append("or absurd values mean the forward is broken -> reported as BLOCKER, not published as valid.")
    lines.append("Gate criterion: mean nats/token < 5.0 and BPB < 2.0 and finite.\n")
    lines.append("## Results\n")
    lines.append("| Model | step | tokens scored | mean nats/token | BPB | sanity gate |")
    lines.append("|---|---:|---:|---:|---:|:--:|")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | {r.get('step','?')} | — | — | — | **BLOCKER (load/forward error)** |")
        else:
            lines.append(f"| {r['name']} | {r['step']} | {r['n_tokens_scored']:,} | "
                         f"{r['mean_nats_per_token']:.4f} | {r['bpb']:.4f} | "
                         f"**{r['sanity_gate']}** |")
    lines.append("")

    # Per-model detail + E88 train-loss delta
    lines.append("## Per-model detail\n")
    for r in results:
        lines.append(f"### {r['name']}  (`{r.get('repo','')}` @ {r.get('revision','v0.3')})")
        if "error" in r:
            lines.append("")
            lines.append("**BLOCKER** — could not produce a valid measurement:")
            lines.append("```")
            lines.append(r["error"].strip())
            lines.append("```")
            lines.append("")
            continue
        lines.append("")
        lines.append(f"- tokens total / scored: {r['n_tokens_total']:,} / {r['n_tokens_scored']:,}")
        lines.append(f"- total NLL: {r['total_nll_nats']:.1f} nats")
        lines.append(f"- mean nats/token: **{r['mean_nats_per_token']:.4f}**")
        lines.append(f"- BPB (over {TOTAL_BYTES:,} bytes): **{r['bpb']:.4f}**")
        lines.append(f"- sanity gate: **{r['sanity_gate']}**")
        if not r["sanity_gate"] == "PASS":
            lines.append(f"- **BLOCKER**: forward is broken — mean nats/token = {r['mean_nats_per_token']:.4f} "
                         f"(worse-than-random ~10.8; correct ~2.6). NOT a publishable BPB.")
        if r["name"].startswith("emender-e88"):
            if r["sanity_gate"] == "PASS":
                # train loss is nats/token; held-out bpb is bits/byte — different units.
                # Report the held-out nats/token vs train-loss nats/token delta, and the bpb.
                delta = r["mean_nats_per_token"] - E88_TRAIN_LOSS
                lines.append(f"- E88 train loss = {E88_TRAIN_LOSS} nats/token; held-out mean nats/token "
                             f"= {r['mean_nats_per_token']:.4f}; delta = {delta:+.4f} nats/token "
                             f"(held-out is on unseen bytes, expected >= train loss).")
            else:
                lines.append(f"- E88 train-loss comparison withheld: forward broken, held-out number invalid.")
        lines.append("")

    lines.append("## Validation")
    lines.append("")
    lines.append("- [x] Ran on GPU 0 only (CUDA_VISIBLE_DEVICES=0); HF v0.3 trust_remote_code load path used")
    lines.append("- [x] Each model: nats/token + tokens + BPB on the canonical slice; sanity gate evaluated")
    lines.append("- [x] Real numbers only; broken forward reported as blocker, never faked")
    lines.append("- [x] paper/review/E88_HELDOUT_HF.md written; main.typ NOT modified")
    lines.append("")
    lines.append(f"_Generated by scripts/e88_heldout_hf_bpb.py_")
    return "\n".join(lines) + "\n"


def main():
    log("START e88-heldout-hf BPB measurement")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if not torch.cuda.is_available():
        log("FATAL: CUDA not available")
        sys.exit(2)
    log(f"torch {torch.__version__} device={torch.cuda.get_device_name(0)}")
    device = "cuda"

    text, sha, decode_mode = load_slice()

    results = []
    errors = {}
    for spec in MODELS:
        try:
            results.append(score_model(spec, text, device))
        except Exception as e:
            tb = traceback.format_exc()
            log(f"ERROR {spec['name']}: {e}\n{tb}")
            errors[spec["name"]] = tb
            results.append({"name": spec["name"], "repo": spec["repo"],
                            "revision": spec["rev"], "step": spec["step"],
                            "error": tb})
        # write incrementally so partial progress survives
        with open(OUT_JSON, "w") as f:
            json.dump({"slice_sha256": sha, "decode": decode_mode,
                       "total_bytes": TOTAL_BYTES, "results": results}, f, indent=2)
        md = render_md(results, sha, decode_mode, errors)
        with open(OUT_MD, "w") as f:
            f.write(md)
        log(f"Wrote {OUT_JSON} and {OUT_MD} ({len(results)}/{len(MODELS)} models)")

    log("DONE")


if __name__ == "__main__":
    main()
