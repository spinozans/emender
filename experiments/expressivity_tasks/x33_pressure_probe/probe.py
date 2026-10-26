#!/usr/bin/env python3
"""X33 — the pressure-predicts-collapse probe (EVAL-ONLY).

Reads the hash-frozen X28E package READ-ONLY (verify_freeze re-verified at
start; nothing under x28e_memory_head is modified), rebuilds each bounded tape
arm from its frozen config, loads the selected checkpoint best.pt from the
sibling results directory, and evaluates the immutable ladder pools with
per-token routing traces collected — the X28E trainer's own _final_evaluation
pattern (microbatching, _forward_binary, logits->preds, binary_report) plus
the pressure instrumentation registered in RECEIPTS.md.

Outputs (per arm, under data/):
  <arm>_per_example.jsonl   per-example scatter (depth, label, pred, pressure)
  <arm>_per_k.json          per-pool per-k aggregates (accuracy, pressure)
  <arm>_case_counts.json    per-pool routing case counts (within word)
  <arm>_series_examples.json  illustrative per-token pressure series
  <arm>_run_receipt.json    freeze/checkpoint identity, pool hashes, walls

No training, no optimizer, no writes outside this experiment directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(
    "/home/erikg/emender/experiments/expressivity_tasks/x28e_memory_head"
).resolve()
REPO_ROOT = PACKAGE_ROOT.parents[2]
RESULTS_ROOT = PACKAGE_ROOT.parent / "results_x28e_memory_head"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from trainer import (  # noqa: E402  (frozen package, read-only import)
    build_tape_model,
    microbatch_plan,
    peak_and_eos_live_from_trace,
    verify_freeze,
)
from generators.stage3_ladder import (  # noqa: E402
    ACCEPT,
    A,
    B,
    BIN_ORDER,
    EOS,
    LADDER_VOCAB,
    PAD,
    SOS,
    binary_report,
    ladder_bin_pool,
    ladder_hash,
    ladder_sweep_pool,
)
from eng.routing_trace import (  # noqa: E402
    ALL_CASES,
    CASE_NAMES,
    NO_WRITE,
    R_CAP,
    R_EMPTY,
    R_NOMATCH,
    R_ZERO_RESIDUAL,
)

WINDOW = 64                      # registered window (RECEIPTS.md)
SAT_NUM = 10                     # saturation: wrcap*10 >= 9*wreq  (>= 0.9)
SAT_DEN = 9
CONSERVATIVE_MIN_REQ = 16        # W64c register minimum window requests
APPEND_CASES = (R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL, R_CAP)

# The B0 sweep points — evaluated on BOTH arms (identical seed-fixed pools).
B0_SWEEP_POINTS = tuple(range(608, 865, 16))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pressure extraction (vectorized) + brute-force oracle (self-test).
# ---------------------------------------------------------------------------
def extract_example_stats(trace, lengths_t: torch.Tensor):
    """Per-example pressure statistics from one microbatch's RoutingTrace.

    Returns a dict of CPU tensors / numpy arrays, each [B] (per-example):
      n_req, n_rcap (head-pooled within word), cum_pressure (pooled),
      cum_pressure_per_head [B, NH], first_sat, first_sat_conservative,
      sust_sat, peak_live_mean_heads, eos_live_mean_heads, n_pop
    All indices are PADDED-SEQUENCE token coordinates; in-word = t < length.
    """
    case = trace.case                    # [B,T,NH] int64
    B, T, NH = case.shape
    dev = case.device
    lengths = lengths_t.to(dev)
    tgrid = torch.arange(T, device=dev)
    in_word = tgrid[None, :] < lengths[:, None]          # [B,T]
    iw3 = in_word[:, :, None]

    req = torch.zeros_like(case, dtype=torch.bool)
    for c in APPEND_CASES:
        req |= case == c
    req &= iw3
    rcap = (case == R_CAP) & iw3

    req_pt = req.sum(-1)                 # [B,T] head-pooled per-token counts
    rcap_pt = rcap.sum(-1)
    n_req = req_pt.sum(1)                # [B]
    n_rcap = rcap_pt.sum(1)
    cum_pressure = n_rcap.float() / n_req.clamp(min=1).float()
    cum_pressure_per_head = rcap.sum(1).float() / req.sum(1).float().clamp(min=1)

    # trailing-64-token window, integer arithmetic (exact >= 0.9 test)
    cumreq = req_pt.cumsum(1)            # [B,T] int64
    cumrcap = rcap_pt.cumsum(1)
    shifted_req = torch.zeros_like(cumreq)
    shifted_rcap = torch.zeros_like(cumrcap)
    if T > WINDOW:
        shifted_req[:, WINDOW:] = cumreq[:, :-WINDOW]
        shifted_rcap[:, WINDOW:] = cumrcap[:, :-WINDOW]
    wreq = cumreq - shifted_req        # requests in (t-63 .. t]
    wrcap = cumrcap - shifted_rcap

    defined = (wreq >= 1) & in_word
    sat = defined & (wrcap * SAT_NUM >= SAT_DEN * wreq)
    sat_c = ((wreq >= CONSERVATIVE_MIN_REQ) & in_word
             & (wrcap * SAT_NUM >= SAT_DEN * wreq))

    def first_index(mask):
        idx = mask.int().argmax(dim=1)
        has = mask.any(dim=1)
        return torch.where(has, idx, torch.full_like(idx, -1)), has

    first_sat, _ = first_index(sat)
    first_sat_c, _ = first_index(sat_c)

    # sustained: every DEFINED register value at t' >= t (within word) >= 0.9.
    # pass[t] = sat[t] OR not-defined[t]; suffix AND of pass.
    pass_mask = sat | ~defined
    suffix_all = torch.flip(
        torch.flip(pass_mask.int(), dims=[1]).cumprod(dim=1), dims=[1]
    ).bool()
    sust_mask = suffix_all & defined
    sust_sat, has_sust = first_index(sust_mask)
    sust_sat = torch.where(has_sust, sust_sat, torch.full_like(sust_sat, -1))

    peaks, eos_live = peak_and_eos_live_from_trace(trace, lengths)
    n_pop = (trace.pop_fired & iw3).sum(dim=(1, 2))

    return {
        "n_req": n_req.cpu().numpy(),
        "n_rcap": n_rcap.cpu().numpy(),
        "cum_pressure": cum_pressure.cpu().numpy(),
        "cum_pressure_per_head": cum_pressure_per_head.cpu().numpy(),
        "first_sat": first_sat.cpu().numpy(),
        "first_sat_conservative": first_sat_c.cpu().numpy(),
        "sust_sat": sust_sat.cpu().numpy(),
        "peak_live": peaks.float().mean(dim=-1).cpu().numpy(),
        "eos_live": eos_live.float().mean(dim=-1).cpu().numpy(),
        "n_pop": n_pop.cpu().numpy(),
    }


def _oracle_example_stats(case_rows, length):
    """Brute-force per-example oracle for ONE example (self-test only):
    case_rows is a list of per-token lists of per-head case codes (padded
    rows beyond the word are ignored). Mirrors RECEIPTS.md definitions
    literally, loop-by-loop."""
    n_req = n_rcap = 0
    per_head_req = [0] * len(case_rows[0])
    per_head_rcap = [0] * len(case_rows[0])
    n_pop = 0
    sat_idx = satc_idx = -1
    for t in range(length):
        for h, c in enumerate(case_rows[t]):
            if c in APPEND_CASES:
                n_req += 1
                per_head_req[h] += 1
                if c == R_CAP:
                    n_rcap += 1
                    per_head_rcap[h] += 1
        # sensitive windowed register
        wreq = wrcap = 0
        for tt in range(max(0, t - WINDOW + 1), t + 1):
            for c in case_rows[tt]:
                if c in APPEND_CASES:
                    wreq += 1
                    if c == R_CAP:
                        wrcap += 1
        if sat_idx < 0 and wreq >= 1 and wrcap * SAT_NUM >= SAT_DEN * wreq:
            sat_idx = t
        if satc_idx < 0 and wreq >= CONSERVATIVE_MIN_REQ \
                and wrcap * SAT_NUM >= SAT_DEN * wreq:
            satc_idx = t
    # sustained oracle
    def defined_at(t):
        wreq = sum(
            1 for tt in range(max(0, t - WINDOW + 1), t + 1)
            for c in case_rows[tt] if c in APPEND_CASES)
        return wreq >= 1

    def sat_at(t):
        wreq = wrcap = 0
        for tt in range(max(0, t - WINDOW + 1), t + 1):
            for c in case_rows[tt]:
                if c in APPEND_CASES:
                    wreq += 1
                    if c == R_CAP:
                        wrcap += 1
        return wreq >= 1 and wrcap * SAT_NUM >= SAT_DEN * wreq
    sust = -1
    for t in range(length):
        ok = all((sat_at(tt) or not defined_at(tt)) for tt in range(t, length))
        if ok and defined_at(t) and sat_at(t):
            sust = t
            break
    cum = n_rcap / n_req if n_req else 0.0
    per_head = [per_head_rcap[h] / per_head_req[h] if per_head_req[h] else 0.0
                for h in range(len(per_head_req))]
    return {"n_req": n_req, "n_rcap": n_rcap, "cum_pressure": cum,
            "cum_pressure_per_head": per_head, "first_sat": sat_idx,
            "first_sat_conservative": satc_idx, "sust_sat": sust}


class _TraceStub:
    def __init__(self, case, pop_fired):
        self.case = case
        self.pop_fired = pop_fired
        self.alloc_target = torch.full_like(case, -1)
        self.freed = torch.zeros(
            (*case.shape, 1), dtype=torch.int8, device=case.device)


def run_self_test() -> int:
    """Tiny-scale brute-force oracle gate (adversarial cases included)."""
    torch.manual_seed(0)
    ok = True
    # case 1 (1 head): one clean append, 9 degraded requests, request-free
    # tail. W64(t) = t/(t+1) for t<=9 -> first saturation at t=9 (9/10);
    # sustained to EOS; conservative register never fires (wreq max 10 < 16).
    rows = [[R_EMPTY]] + [[R_CAP]] * 9 + [[NO_WRITE]] * 2
    assert len(rows) == 12
    case = torch.tensor([[r for r in rows]], dtype=torch.int64)
    lengths = torch.tensor([12])
    stub = _TraceStub(case, torch.zeros_like(case, dtype=torch.bool))
    got = extract_example_stats(stub, lengths)
    want = _oracle_example_stats(rows, 12)
    checks = [
        ("n_req", got["n_req"][0], want["n_req"]),
        ("n_rcap", got["n_rcap"][0], want["n_rcap"]),
        ("cum_pressure", float(got["cum_pressure"][0]), want["cum_pressure"]),
        ("per_head", got["cum_pressure_per_head"][0].tolist(),
         want["cum_pressure_per_head"]),
        ("first_sat", int(got["first_sat"][0]), want["first_sat"]),
        ("first_sat_c", int(got["first_sat_conservative"][0]),
         want["first_sat_conservative"]),
        ("sust_sat", int(got["sust_sat"][0]), want["sust_sat"]),
    ]
    # case 2 (adversarial): PAD region carries junk R-CAP — must be EXCLUDED;
    # in-word suffix must not be broken by undefined out-of-word windows.
    rows2 = [[R_EMPTY], [R_EMPTY], [R_CAP], [R_CAP], [R_CAP],
             [NO_WRITE], [NO_WRITE]] + [[R_CAP]] * 3   # last 3 = PAD junk
    case2 = torch.tensor([[r for r in rows2]], dtype=torch.int64)
    lengths2 = torch.tensor([7])
    stub2 = _TraceStub(case2, torch.zeros_like(case2, dtype=torch.bool))
    got2 = extract_example_stats(stub2, lengths2)
    want2 = _oracle_example_stats(rows2, 7)
    checks += [
        ("2.n_rcap", int(got2["n_rcap"][0]), want2["n_rcap"]),
        ("2.n_req", int(got2["n_req"][0]), want2["n_req"]),
        ("2.cum_pressure", float(got2["cum_pressure"][0]),
         want2["cum_pressure"]),
        ("2.first_sat", int(got2["first_sat"][0]), want2["first_sat"]),
        ("2.sust_sat", int(got2["sust_sat"][0]), want2["sust_sat"]),
    ]
    # case 3 (adversarial): a single defined-but-clean window mid-word
    # (requests, zero rcap) must break "sustained" for earlier t only via
    # defined positions; undefined gaps must NOT break it.
    rows3 = [[R_CAP], [R_CAP], [NO_WRITE], [NO_WRITE], [R_CAP], [R_CAP]]
    case3 = torch.tensor([[r for r in rows3]], dtype=torch.int64)
    lengths3 = torch.tensor([6])
    stub3 = _TraceStub(case3, torch.zeros_like(case3, dtype=torch.bool))
    got3 = extract_example_stats(stub3, lengths3)
    want3 = _oracle_example_stats(rows3, 6)
    checks += [
        ("3.first_sat", int(got3["first_sat"][0]), want3["first_sat"]),
        ("3.sust_sat", int(got3["sust_sat"][0]), want3["sust_sat"]),
        ("3.cum_pressure", float(got3["cum_pressure"][0]),
         want3["cum_pressure"]),
    ]
    # case 5: long a-phase-like run — BOTH registers fire (wreq grows past
    # 16; W(t)=t-1/t... with 2 clean appends then 23 rcap: saturation at
    # t=19 with wreq=20).
    rows5 = [[R_EMPTY]] * 2 + [[R_CAP]] * 23
    case5 = torch.tensor([[r for r in rows5]], dtype=torch.int64)
    stub5 = _TraceStub(case5, torch.zeros_like(case5, dtype=torch.bool))
    got5 = extract_example_stats(stub5, torch.tensor([25]))
    want5 = _oracle_example_stats(rows5, 25)
    checks += [
        ("5.first_sat", int(got5["first_sat"][0]), want5["first_sat"]),
        ("5.first_sat_c", int(got5["first_sat_conservative"][0]),
         want5["first_sat_conservative"]),
        ("5.sust_sat", int(got5["sust_sat"][0]), want5["sust_sat"]),
        ("5.cum_pressure", float(got5["cum_pressure"][0]),
         want5["cum_pressure"]),
    ]
    # case 4: zero requests anywhere -> pressure 0, no events
    rows4 = [[NO_WRITE]] * 5
    case4 = torch.tensor([[r for r in rows4]], dtype=torch.int64)
    stub4 = _TraceStub(case4, torch.zeros_like(case4, dtype=torch.bool))
    got4 = extract_example_stats(stub4, torch.tensor([5]))
    checks += [
        ("4.cum_pressure", float(got4["cum_pressure"][0]), 0.0),
        ("4.first_sat", int(got4["first_sat"][0]), -1),
        ("4.sust_sat", int(got4["sust_sat"][0]), -1),
    ]
    def _same(a, b):
        if isinstance(a, float):
            return abs(a - b) < 1e-6
        if isinstance(a, list):
            return len(a) == len(b) and all(
                abs(x - y) < 1e-6 if isinstance(x, float) else x == y
                for x, y in zip(a, b))
        return a == b
    for name, a, b in checks:
        same = _same(a, b)
        print(f"  self-test {name}: got={a} want={b} {'OK' if same else 'FAIL'}")
        ok &= bool(same)
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Pools and per-example evaluation.
# ---------------------------------------------------------------------------
def pools_for_arm(arm: str, arm_cfg: dict) -> dict:
    pools = {name: ladder_bin_pool(name) for name in BIN_ORDER}
    own = arm_cfg["fine_sweep"]["points"]
    for k in own:
        pools[f"sweep_k{k}"] = ladder_sweep_pool(k)
    if arm != "tape_B0":                       # the same-k comparison set
        for k in B0_SWEEP_POINTS:
            pools[f"sweep_k{k}"] = ladder_sweep_pool(k)
    return pools


def classify_shape(row: np.ndarray, length: int) -> str:
    """Reject-shape classifier from the token row (leading_b / count_mismatch
    / trailing_a; accepts return 'balanced')."""
    word = row[1:length - 1]                    # strip SOS/EOS
    if word.size == 0:
        return "empty"
    has_a = bool((word == A).any())
    has_b = bool((word == B).any())
    if not has_a:
        return "leading_b" if has_b else "empty"
    if word[0] == B:
        return "leading_b"
    # first B position
    fb = int(np.argmax(word == B)) if has_b else len(word)
    if not has_b:
        return "count_mismatch"                # a^j only (j != k impossible here)
    if (word[fb:] == A).any():
        return "trailing_a"
    # a^j b^k with j != k -> count mismatch; j == k -> balanced
    j = fb
    k = len(word) - fb
    return "balanced" if j == k else "count_mismatch"


SERIES_POOLS = {f"sweep_k{k}" for k in (720, 736, 752, 768, 864)} | {"bin_9x"}
SERIES_DEPTH_TARGETS = {1025, 1088, 1152}


def evaluate_arm(arm: str, device: str, limit_pools=None):
    import torch  # noqa: F401  (already imported at module top)

    manifest_hash = json.loads(
        (PACKAGE_ROOT / "freeze_manifest.json").read_text())["manifest_hash"]
    manifest = verify_freeze(manifest_hash)
    print(f"[{arm}] freeze verified: {manifest['manifest_hash']} "
          f"({len(manifest['files'])} files)")

    arm_cfg = json.loads((PACKAGE_ROOT / "configs" / f"{arm}.json").read_text())
    torch.manual_seed(0)
    np.random.seed(0)
    model = build_tape_model(arm_cfg).to(device)
    model.attach_binary_stage(LADDER_VOCAB)
    model.to(device)   # fresh stage-local embedding -> device (trainer order)
    ckpt = RESULTS_ROOT / f"{arm}_seed0" / "best.pt"
    state = torch.load(ckpt, map_location=device)
    # best.pt was saved during stage 3 with the stage-2 head_multi still
    # attached (attach_binary_stage replaces only the embedding); tolerate
    # EXACTLY those two stale keys and nothing else.
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing}"
    assert sorted(unexpected) == ["head_multi.bias", "head_multi.weight"], (
        f"unexpected keys: {unexpected}")
    model.eval()
    ckpt_sha = sha256_file(ckpt)

    micro = arm_cfg["microbatch"]
    pools = pools_for_arm(arm, arm_cfg)
    if limit_pools:
        pools = {k: v for k, v in pools.items() if k in limit_pools}

    per_example_path = DATA / f"{arm}_per_example.jsonl"
    per_k: dict = {}
    case_counts: dict = {}
    series_examples: list = []
    walls: dict = {}
    bin_crosscheck: dict = {}
    t_arm0 = time.time()

    with per_example_path.open("w") as exf, torch.no_grad():
        for name, (inputs, labels, lengths, meta) in pools.items():
            t0 = time.time()
            depths = np.asarray(meta["depths"])
            preds = np.empty(len(labels), dtype=np.int64)
            stats = {k: [] for k in (
                "n_req", "n_rcap", "cum_pressure", "cum_pressure_per_head",
                "first_sat", "first_sat_conservative", "sust_sat",
                "peak_live", "eos_live", "n_pop")}
            pool_case = {CASE_NAMES[c]: 0 for c in ALL_CASES}
            n_mb = 0
            for mb in microbatch_plan(lengths, micro["token_budget"],
                                      micro["max_sequences"]):
                t_pad = int(lengths[mb].max())
                x = torch.from_numpy(np.ascontiguousarray(
                    inputs[mb][:, :t_pad])).to(device)
                ln = torch.from_numpy(np.ascontiguousarray(lengths[mb])).to(device)
                y, _, state_out, trace = model.forward(x, collect_trace=True)
                idx = torch.arange(x.shape[0], device=device)
                logits = model.head_binary(y[idx, ln - 1])
                if not bool(torch.isfinite(logits).all()):
                    raise SystemExit(
                        f"[{arm}] {name}: nonfinite eval logits — abort")
                preds[mb] = (logits[:, 1] > logits[:, 0]).long().cpu().numpy()
                st = extract_example_stats(trace, ln)
                for k in stats:
                    stats[k].append(st[k])
                # pool case counts within word
                in_word = (torch.arange(t_pad, device=device)[None, :]
                           < ln[:, None])
                for c in ALL_CASES:
                    pool_case[CASE_NAMES[c]] += int(
                        ((trace.case == c) & in_word[:, :, None]).sum())
                # illustrative per-token series for the first accept example
                if name in SERIES_POOLS:
                    want_k = (
                        depths[mb][0] in SERIES_DEPTH_TARGETS
                        if name == "bin_9x" else True)
                    acc_local = np.where(
                        labels[mb] == ACCEPT)[0]
                    if want_k and len(acc_local) and not any(
                            s["pool"] == name for s in series_examples):
                        i0 = int(acc_local[0])
                        req_tok = (
                            (trace.case == R_EMPTY)
                            | (trace.case == R_NOMATCH)
                            | (trace.case == R_ZERO_RESIDUAL)
                            | (trace.case == R_CAP))
                        req_pt = (req_tok & in_word[:, :, None]).sum(-1)[i0]
                        rcap_pt = ((trace.case == R_CAP)
                                   & in_word[:, :, None]).sum(-1)[i0]
                        L = int(lengths[mb][i0])
                        series_examples.append({
                            "pool": name, "k": int(depths[mb][i0]),
                            "label": int(labels[mb][i0]),
                            "pred": int(preds[mb][i0]),
                            "req_per_token": req_pt[:L].cpu().tolist(),
                            "rcap_per_token": rcap_pt[:L].cpu().tolist(),
                        })
                n_mb += 1
                del trace, y, state_out
            walls[name] = round(time.time() - t0, 2)

            # flatten per-example stats
            flat = {k: np.concatenate(v, axis=0) if v else np.array([])
                    for k, v in stats.items()}
            correct = preds == labels
            rep = binary_report(labels, preds)
            case_counts[name] = pool_case

            # per-k aggregation
            for k in sorted(set(depths.tolist())):
                sel = depths == k
                acc_k = binary_report(labels[sel], preds[sel])
                acc_sel = sel & (labels == ACCEPT)
                cp = flat["cum_pressure"]
                entry = {
                    "k": int(k), "n": int(sel.sum()),
                    "balanced_accuracy": acc_k["balanced_accuracy"],
                    "n_accept": int(acc_sel.sum()),
                    "mean_cum_pressure_accepts": (
                        float(cp[acc_sel].mean()) if acc_sel.any() else None),
                    "mean_cum_pressure_rejects": (
                        float(cp[sel & (labels != ACCEPT)].mean())
                        if (sel & (labels != ACCEPT)).any() else None),
                    "frac_accepts_pressure_ge_0.5": (
                        float((cp[acc_sel] >= 0.5).mean())
                        if acc_sel.any() else None),
                    "frac_accepts_pressure_ge_0.9": (
                        float((cp[acc_sel] >= 0.9).mean())
                        if acc_sel.any() else None),
                    "frac_accepts_sat_event": (
                        float((flat["first_sat"][acc_sel] >= 0).mean())
                        if acc_sel.any() else None),
                    "median_first_sat_accepts": (
                        int(np.median(flat["first_sat"][acc_sel]))
                        if acc_sel.any() else None),
                    "mean_peak_live": float(flat["peak_live"][sel].mean()),
                }
                per_k.setdefault(name, {})[str(k)] = entry

            # per-example records
            for i in range(len(labels)):
                k = int(depths[i])
                first_sat = int(flat["first_sat"][i])
                length_i = int(lengths[i])
                b_start = k + 1 if labels[i] == ACCEPT else None
                rec = {
                    "pool": name, "k": k, "label": int(labels[i]),
                    "pred": int(preds[i]), "correct": bool(correct[i]),
                    "length": length_i,
                    "shape": (classify_shape(inputs[i], length_i)
                              if labels[i] != ACCEPT else "balanced"),
                    "n_req": int(flat["n_req"][i]),
                    "n_rcap": int(flat["n_rcap"][i]),
                    "cum_pressure": float(flat["cum_pressure"][i]),
                    "cum_pressure_per_head": [
                        round(float(v), 6)
                        for v in flat["cum_pressure_per_head"][i]],
                    "first_sat": first_sat,
                    "first_sat_conservative": int(
                        flat["first_sat_conservative"][i]),
                    "sust_sat": int(flat["sust_sat"][i]),
                    "peak_live": float(flat["peak_live"][i]),
                    "eos_live": float(flat["eos_live"][i]),
                    "n_pop": int(flat["n_pop"][i]),
                }
                if labels[i] == ACCEPT:
                    rec["b_start"] = b_start
                    rec["eos_idx"] = length_i - 1
                    if first_sat >= 0:
                        rec["lead_b"] = b_start - first_sat
                        rec["lead_eos"] = (length_i - 1) - first_sat
                    else:
                        rec["lead_b"] = None
                        rec["lead_eos"] = None
                exf.write(json.dumps(rec) + "\n")

            line = (f"[{arm}] {name}: acc={rep['balanced_accuracy']:.4f} "
                    f"recalls={ {a: round(b, 4) for a, b in rep['per_class_recall'].items()} } "
                    f"meanP_acc={flat['cum_pressure'][labels == ACCEPT].mean():.4f} "
                    f"peak={flat['peak_live'].mean():.1f} "
                    f"({n_mb} mb, {walls[name]}s)")
            print(line, flush=True)
            if name in BIN_ORDER:
                bin_crosscheck[name] = rep

    # byte-identity gate vs the X28E final_eval receipts (same pools, same
    # checkpoint, deterministic eval): balanced accuracy must match EXACTLY.
    final = json.loads(
        (RESULTS_ROOT / f"{arm}_seed0" / "final_eval.json").read_text())
    for name, rep in bin_crosscheck.items():
        want = final[name]["report"]["balanced_accuracy"]
        got = rep["balanced_accuracy"]
        if abs(got - want) > 1e-9:
            raise SystemExit(
                f"[{arm}] {name}: balanced accuracy {got} != X28E receipt "
                f"{want} — evaluation identity broken, abort")
        print(f"[{arm}] crosscheck {name}: {got:.4f} == receipt {want:.4f} OK")

    (DATA / f"{arm}_per_k.json").write_text(json.dumps(per_k, indent=1))
    (DATA / f"{arm}_case_counts.json").write_text(
        json.dumps(case_counts, indent=1))
    (DATA / f"{arm}_series_examples.json").write_text(
        json.dumps(series_examples, indent=1))
    receipt = {
        "arm": arm,
        "freeze_manifest_hash": manifest_hash,
        "checkpoint": f"results_x28e_memory_head/{arm}_seed0/best.pt",
        "checkpoint_sha256": ckpt_sha,
        "device": device,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0)
        if device.startswith("cuda") else "cpu",
        "pools": {name: ladder_hash(name, *pools[name][:3])
                  for name in pools},
        "walls_seconds": walls,
        "total_wall_seconds": round(time.time() - t_arm0, 1),
        "note": "eval-only; frozen package read-only; no optimizer",
    }
    (DATA / f"{arm}_run_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"[{arm}] DONE in {receipt['total_wall_seconds']}s "
          f"(example records: {per_example_path.name})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["tape_B0", "tape_B0x4"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--limit-pools", nargs="*", default=None,
                    help="restrict to named pools (smoke)")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        return run_self_test()
    if not args.arm:
        ap.error("--arm required (or --self-test)")
    return evaluate_arm(args.arm, args.device, args.limit_pools)


if __name__ == "__main__":
    raise SystemExit(main())
