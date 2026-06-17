#!/usr/bin/env python3
"""Re-run offline held-out BPB on the FUSED kernel for both references.

task: re-run-offline.

Background: the committed offline-eval-references emender curve was produced on
the EAGER per-step PyTorch recurrence. The E88/E97 fused Triton paths in
ndm/models/e88_fla_hybrid.py were gated on ``self.training``, so under
``model.eval()`` (the offline scorer) the emender E97 recurrence silently fell
back to the eager scan -- a NON-NEGOTIABLE #1 violation. capability-track-
references fixed this with an opt-in ``fused_inference`` gate that
scripts/eval_checkpoint.py:build_model auto-enables. gdn2-mlp was already fused
(FLA chunk, training-independent).

This driver regenerates the committed curve on the now-fused loader:

  * Same checkpoints (the original committed step set), so the fused-vs-eager
    delta is row-matched.
  * Same shared held-out tensor (md5 8e1198ab...), same y-mode ``train`` swap,
    forward-only, bf16.
  * A REAL kernel-invocation guard (NOT the use_triton config flag): the actual
    fused-kernel entry points are wrapped with call counters and the eager
    per-step sentinel (E88FLAHybrid._apply_state_activation, called only inside
    the eager time loop) is wrapped too. Per emender checkpoint we assert
    fused_calls > 0 AND eager_calls == 0.

Reuses scripts/eval_checkpoint.py's build_model / load / score helpers so the
scored path is byte-identical to the committed tool, differing only in the
(now auto-enabled) fused inference gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_checkpoint as ec  # noqa: E402

HERE = Path(__file__).resolve().parent
HELDOUT = Path(
    "/mnt/nvme1n1/erikg/ref_emender_mlp.contaminated_2057/"
    "heldout_pile_tail_p50k_2048_1m.pt"
)
EMENDER_DIR = Path(
    "/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750"
)
GDN2_DIR = Path(
    "/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627"
)

# The EXACT checkpoint step sets in the committed CSVs, so the fused re-score is
# a row-matched eager->fused comparison (training has since produced additional
# checkpoints; we deliberately re-score only the original set for a clean delta).
EMENDER_STEPS = [21500, 43000, 64500, 86000, 107500]
GDN2_STEPS = [25000, 50000, 75000, 100000, 125000]

CSV_FIELDS = ec.CSV_FIELDS  # step, tokens, ce, bpb, split, checkpoint

# ---------------------------------------------------------------------------
# Kernel-invocation instrumentation (NOT the use_triton config flag).
# ---------------------------------------------------------------------------
COUNTERS = {
    "fused_seq": 0,        # e88_triton_optimized_apply  (emender E97 sequential)
    "fused_chunked": 0,    # e97_delta_chunked_triton    (emender E97 chunked)
    "eager": 0,            # E88FLAHybrid._apply_state_activation (eager time loop)
    "gdn2_chunk": 0,       # FLA chunk_gated_delta_rule   (gdn2-mlp)
    "gdn2_fused_recurrent": 0,  # FLA fused_recurrent_gated_delta_rule (gdn2-mlp)
}


def _wrap(mod, attr, key):
    """Wrap mod.attr with a COUNTERS[key] increment; return True on success."""
    if not hasattr(mod, attr):
        return False
    orig = getattr(mod, attr)

    def wrapper(*args, **kwargs):
        COUNTERS[key] += 1
        return orig(*args, **kwargs)

    # Use a UNIQUE marker (not functools' __wrapped__) for the already-wrapped
    # check: kernel entry points like chunk_gdn2 are themselves wrapped with
    # functools.wraps (autocast/custom_fwd) and ALREADY carry __wrapped__, so a
    # __wrapped__-based skip-guard would never wrap them.
    wrapper._kernel_inv_wrapped = True
    wrapper.__wrapped__ = orig
    setattr(mod, attr, wrapper)
    return True


def install_emender_instrumentation():
    """Wrap the emender fused kernels + the eager sentinel.

    The forward does `from ndm.triton... import <fn>` at CALL time, so patching
    the module attribute here is picked up by the lazy import. The eager
    sentinel is a method called as self._apply_state_activation(...) only inside
    the per-step PyTorch time loop.
    """
    import ndm.triton.e88_triton_optimized as seqmod
    import ndm.triton.e97_chunked_autograd as chmod
    from ndm.models.e88_fla_hybrid import E88FLAHybrid

    ok_seq = _wrap(seqmod, "e88_triton_optimized_apply", "fused_seq")
    ok_ch = _wrap(chmod, "e97_delta_chunked_triton", "fused_chunked")

    orig_eager = E88FLAHybrid._apply_state_activation

    def eager_wrapper(self, pre):
        COUNTERS["eager"] += 1
        return orig_eager(self, pre)

    eager_wrapper.__wrapped__ = orig_eager
    E88FLAHybrid._apply_state_activation = eager_wrapper
    return {"fused_seq": ok_seq, "fused_chunked": ok_ch, "eager": True}


def install_gdn2_instrumentation(model=None):
    """Count the external GDN-2 fused kernel launches (chunk_gdn2 /
    fused_recurrent_gdn2) for gdn2-mlp.

    The external GatedDeltaNet-2 checkout is loaded under a custom module name
    (e.g. _external_gdn2_lit_gpt.gdn2) and the layer calls the bare names
    `chunk_gdn2` / `fused_recurrent_gdn2` resolved as module globals. So we scan
    every loaded module for those symbols and wrap each in place -- a real
    kernel-invocation counter, not the mode/use_triton config flag. (FLA's
    chunk_gated_delta_rule is NOT what this checkout uses.)
    """
    patched = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        for attr, key in (("chunk_gdn2", "gdn2_chunk"),
                          ("fused_recurrent_gdn2", "gdn2_fused_recurrent")):
            fn = getattr(mod, attr, None)
            if callable(fn) and not getattr(fn, "_kernel_inv_wrapped", False):
                _wrap(mod, attr, key)
                patched.append(f"{mod_name}.{attr}")
    return patched


# ---------------------------------------------------------------------------
# Scoring (mirrors eval_checkpoint.main's per-checkpoint loop).
# ---------------------------------------------------------------------------
def ckpt_path_for_step(run_dir: Path, step: int) -> Path:
    matches = sorted(run_dir.glob(f"checkpoint_step_{step:06d}_*.pt"))
    if not matches:
        matches = sorted(run_dir.glob(f"checkpoint_step_{step:06d}.pt"))
    if not matches:
        raise FileNotFoundError(f"no checkpoint for step {step} in {run_dir}")
    return matches[0].resolve()


def score_one(model_name, ckpt_path, scoring, device, batch_size, guard_log):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{ckpt_path} has no model_state_dict")
    cfg = ec.checkpoint_args(ckpt_path, checkpoint, None)
    model_args = ec.namespace_from_config(cfg)
    step = ec.checkpoint_step(ckpt_path, checkpoint)
    model = ec.build_model(model_args, device)
    swapped = ec.load_checkpoint_weights(model, checkpoint, model_args, "train")

    if model_name == "gdn2":
        patched = install_gdn2_instrumentation(model)
        if patched:
            print(f"[rerun_fused] gdn2 kernel instrumentation: {patched}", flush=True)

    # Reset counters and score.
    before = dict(COUNTERS)
    ce, bpb, scored = ec.score_tensor(
        model=model,
        scoring=scoring,
        device=device,
        batch_size=batch_size,
        use_bf16=bool(getattr(model_args, "bf16", False)),
    )
    delta = {k: COUNTERS[k] - before[k] for k in COUNTERS}

    # --- the real kernel-invocation guard ---
    if model_name == "emender":
        fused = delta["fused_seq"] + delta["fused_chunked"]
        eager = delta["eager"]
        if not (fused > 0 and eager == 0):
            raise RuntimeError(
                f"[fused-guard] FAIL emender step={step}: fused_calls={fused} "
                f"(seq={delta['fused_seq']} chunked={delta['fused_chunked']}) "
                f"eager_calls={eager}; expected fused>0 AND eager==0"
            )
        guard_status = (
            f"PASS fused={fused} (seq={delta['fused_seq']} "
            f"chunked={delta['fused_chunked']}) eager={eager}"
        )
    else:  # gdn2: FLA chunk is training-independent; no E88 eager loop exists.
        gdn2_fused = delta["gdn2_chunk"] + delta["gdn2_fused_recurrent"]
        eager = delta["eager"]
        if eager != 0:
            raise RuntimeError(
                f"[fused-guard] FAIL gdn2 step={step}: eager E88 loop ran "
                f"({eager}) -- gdn2 must never hit the E88 eager path"
            )
        guard_status = (
            f"PASS gdn2_fla_fused={gdn2_fused} "
            f"(chunk={delta['gdn2_chunk']} fused_recurrent={delta['gdn2_fused_recurrent']}) "
            f"eager={eager}"
        )

    n_e88 = sum(
        1 for _ in model.modules()
        if type(_).__name__ == "E88FLAHybrid"
    )
    print(
        f"[rerun_fused] {model_name} step={step} ce={ce:.8f} bpb={bpb:.8f} "
        f"y_swap={swapped} n_e88={n_e88} guard={guard_status}",
        flush=True,
    )
    guard_log.append(
        {
            "model": model_name,
            "step": step,
            "ce": ce,
            "bpb": bpb,
            "y_swap": swapped,
            "n_e88_layers": n_e88,
            "kernel_calls": delta,
            "guard": guard_status,
            "checkpoint": str(ckpt_path),
        }
    )

    row = {
        "step": step,
        "tokens": ec.tokens_at_step(step, model_args, None),
        "ce": f"{ce:.8f}",
        "bpb": f"{bpb:.8f}",
        "split": "primary",
        "checkpoint": str(ckpt_path),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def write_csv(path: Path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[rerun_fused] wrote {path} ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--models", default="emender,gdn2",
        help="Comma list of models to (re)score: emender,gdn2 (default both).",
    )
    args = ap.parse_args()
    want = {m.strip() for m in args.models.split(",") if m.strip()}

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(args.device)

    instr = install_emender_instrumentation()
    print(f"[rerun_fused] emender instrumentation: {instr}", flush=True)

    scoring = ec.load_scoring_tensor(HELDOUT)
    print(
        f"[rerun_fused] held-out tensor {HELDOUT.name} "
        f"chunks={tuple(scoring.chunks.shape)} bpt={scoring.bytes_per_token}",
        flush=True,
    )

    guard_log = []
    jobs = [
        ("emender", EMENDER_DIR, EMENDER_STEPS),
        ("gdn2", GDN2_DIR, GDN2_STEPS),
    ]
    for name, run_dir, steps in jobs:
        if name not in want:
            continue
        rows = []
        for step in steps:
            ckpt = ckpt_path_for_step(run_dir, step)
            rows.append(score_one(name, ckpt, scoring, device, args.batch_size, guard_log))
        write_csv(HERE / f"{name}_heldout_bpb.fused.csv", rows)

    # Merge into fused_guard.json: replace entries for the models we just ran,
    # keep entries for models we skipped (so a gdn2-only re-run preserves emender).
    guard_path = HERE / "fused_guard.json"
    existing = []
    if guard_path.exists():
        try:
            existing = json.loads(guard_path.read_text())
        except Exception:
            existing = []
    merged = [e for e in existing if e.get("model") not in want] + guard_log
    merged.sort(key=lambda e: (e.get("model", ""), e.get("step", 0)))
    guard_path.write_text(json.dumps(merged, indent=2))
    print(f"[rerun_fused] wrote {guard_path}", flush=True)
    print("[rerun_fused] DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
