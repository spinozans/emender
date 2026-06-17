#!/usr/bin/env python3
"""Capability-vs-token tracking on the emender / gdn2 reference checkpoints.

Task: ``capability-track-references``. The paired offline LM-BPB curve
(``results/offline-eval-references``) shows gdn2 leads emender by ~0.13-0.19 BPB
at matched tokens -- the expected convergent-loss signal. LM loss does not
distinguish architectures on *capability*. This script tracks the capability
axis instead: it scores the SAME multiple-choice reasoning/QA panel that sits
behind paper Fig 3 / ``paper/results/qa_reasoning`` on every saved checkpoint of
both references, at the checkpoint token cadence, so we can ask whether emender
DIVERGES from gdn2 on any capability axis as tokens grow or converges there too.

It deliberately stitches together the two already-trusted pieces:

  * Model loading -- reused verbatim from ``scripts/eval_checkpoint.py`` (the
    loader the offline-eval-references BPB task used): builds the FUSED kernel
    (emender E97 ``use_triton=1`` hard-imports Triton, no eager fallback; gdn2
    ``gdn2-mlp`` FLA chunked GDN-2), and applies the schedule-free y-mode weight
    swap (optimizer.train()) when ``--y-mode train``.
  * Scoring -- reused verbatim from ``scripts/racer_eval_suite.py``: continuation
    NLL over tokenized choices, forward-only (``model(x, return_loss=False)``,
    under ``torch.no_grad``), ``avg_nll`` primary score.

Outputs a long-form per-(checkpoint x category) CSV plus a per-item JSONL so
accuracies carry honest binomial / bootstrap confidence intervals downstream.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "ndm" / "cuda"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --- proven loader (fused + schedule-free y-mode swap) ----------------------
import eval_checkpoint as ckpt_mod  # noqa: E402

# --- proven scorer (forward-only continuation NLL) --------------------------
import racer_eval_suite as racer  # noqa: E402


CSV_FIELDS = [
    "model", "panel", "level", "step", "tokens", "params", "sf_y_swap",
    "score_mode", "category", "n", "correct", "accuracy",
]


def log(msg: str) -> None:
    print(f"[capability_track] {msg}", flush=True)


def assert_fused_recurrence(model, device) -> str:
    """Real FUSED-GUARD: instrument the kernel entry points and run a tiny eval
    forward, asserting the recurrence executed the FUSED Triton kernel and NOT
    the eager per-step PyTorch scan (NON-NEGOTIABLE #1). Unlike a config-flag
    check, this catches the train-vs-eval eager fallback (the fused E88/E97 paths
    are gated on self.training). No-op assertion for archs without E88FLAHybrid
    mixers (gdn2-mlp is fused via FLA chunk regardless of mode)."""
    try:
        from ndm.models.e88_fla_hybrid import E88FLAHybrid
    except Exception:
        return "no-e88-mixer"
    mixers = [m for m in model.modules() if isinstance(m, E88FLAHybrid)]
    if not mixers:
        return "no-e88-mixer (fused arch is FLA-chunk, training-independent)"
    import ndm.triton.e88_triton_optimized as topt
    import ndm.triton.e97_chunked_autograd as tchunk
    counts = {"fused_seq": 0, "fused_chunked": 0, "eager_act": 0}
    o_seq, o_chunk = topt.e88_triton_optimized_apply, tchunk.e97_delta_chunked_triton
    o_act = E88FLAHybrid._apply_state_activation

    def w_seq(*a, **k):
        counts["fused_seq"] += 1
        return o_seq(*a, **k)

    def w_chunk(*a, **k):
        counts["fused_chunked"] += 1
        return o_chunk(*a, **k)

    def w_act(self, pre):
        counts["eager_act"] += 1
        return o_act(self, pre)

    topt.e88_triton_optimized_apply = w_seq
    tchunk.e97_delta_chunked_triton = w_chunk
    E88FLAHybrid._apply_state_activation = w_act
    try:
        was_training = model.training
        model.eval()
        x = torch.randint(0, 256, (1, 16), device=device)
        with torch.no_grad():
            model(x, return_loss=False)
        if was_training:
            model.train()
    finally:
        topt.e88_triton_optimized_apply = o_seq
        tchunk.e97_delta_chunked_triton = o_chunk
        E88FLAHybrid._apply_state_activation = o_act
    fused = counts["fused_seq"] + counts["fused_chunked"]
    if fused == 0 or counts["eager_act"] > 0:
        raise RuntimeError(
            "[fused-guard] E88/E97 recurrence ran EAGER at eval "
            f"(fused={fused}, eager_act={counts['eager_act']}) -- NO eager fallback "
            "allowed (NON-NEGOTIABLE #1). Need fused_inference=True on the mixers.")
    return (f"fused-guard PASS: {len(mixers)} mixers, "
            f"fused_seq={counts['fused_seq']} fused_chunked={counts['fused_chunked']} "
            f"eager_act=0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", action="append", default=[], metavar="LABEL=RUN_DIR",
        help="Model label and run-dir, e.g. emender=/path/to/runs/levelE97_...",
    )
    p.add_argument(
        "--panel", action="append", default=[], metavar="NAME=JSONL",
        help="Panel name and probe JSONL, e.g. knowledge=panels/knowledge.jsonl",
    )
    p.add_argument("--glob", default="checkpoint_step_*.pt")
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--out-items", required=True, type=Path,
                   help="JSONL of per-item correctness for CIs / bootstrap.")
    p.add_argument("--y-mode", choices=["train", "eval", "saved"], default="train")
    p.add_argument("--batch-size", type=int,
                   default=int(os.environ.get("CAP_EVAL_BS", "16")))
    p.add_argument("--device", default="cuda")
    p.add_argument("--primary-score", choices=["avg_nll", "nll"], default="avg_nll")
    p.add_argument("--no-lease", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    return p.parse_args()


def split_kv(spec: str, what: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--{what} expects LABEL=VALUE, got {spec!r}")
    label, value = spec.split("=", 1)
    if not label or not value:
        raise SystemExit(f"--{what} expects non-empty LABEL=VALUE, got {spec!r}")
    return label, value


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_items(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_keys(path: Path) -> set[tuple[str, str, int]]:
    """(model, panel, step) tuples already present, for idempotent resume."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    seen: set[tuple[str, str, int]] = set()
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                seen.add((r["model"], r["panel"], int(r["step"])))
            except (KeyError, ValueError):
                continue
    return seen


def load_panel(name: str, jsonl: str) -> list[racer.EvalItem]:
    items = racer.normalize_probe_rows(racer.load_probe_records(jsonl))
    if not items:
        raise SystemExit(f"panel {name} ({jsonl}) loaded zero items")
    return items


def score_one(model, items, enc, device, batch_size, score_mode, primary):
    """Forward-only continuation-NLL scoring; returns (summary, rows)."""
    score_keys = ["nll", "avg_nll"]
    rows = racer.score_items(
        model=model,
        items=items,
        enc=enc,
        device=device,
        pad_token=0,
        batch_size=batch_size,
        score_mode=score_mode,
        neutral_prefix="\x1e",
        compute_pmi=False,
    )
    summary = racer.summarize_rows(rows, score_keys)
    return summary, rows


def main() -> int:
    args = parse_args()
    ckpt_mod.maybe_reexec_with_gpu_lease(sys.argv[1:], args.no_lease)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    models = [split_kv(s, "model") for s in args.model]
    panels_spec = [split_kv(s, "panel") for s in args.panel]
    if not models:
        raise SystemExit("at least one --model LABEL=RUN_DIR required")
    if not panels_spec:
        raise SystemExit("at least one --panel NAME=JSONL required")

    panels = [(name, load_panel(name, jsonl)) for name, jsonl in panels_spec]
    for name, items in panels:
        log(f"panel {name}: {len(items)} items")

    already = done_keys(args.out_csv)

    import tiktoken

    for label, run_dir in models:
        run_path = Path(run_dir)
        ckpt_paths = sorted(run_path.glob(args.glob))
        if not ckpt_paths:
            log(f"WARNING no checkpoints under {run_path}/{args.glob}")
            continue
        for ckpt_path in ckpt_paths:
            ckpt_path = ckpt_path.resolve()
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "model_state_dict" not in checkpoint:
                    raise KeyError(f"{ckpt_path} has no model_state_dict")
                cfg = ckpt_mod.checkpoint_args(ckpt_path, checkpoint, None)
                model_args = ckpt_mod.namespace_from_config(cfg)
                step = ckpt_mod.checkpoint_step(ckpt_path, checkpoint)

                pending = [(n, it) for n, it in panels if (label, n, step) not in already]
                if not pending:
                    log(f"skip {label} step={step} (all panels scored)")
                    continue

                model = ckpt_mod.build_model(model_args, device)
                swapped = ckpt_mod.load_checkpoint_weights(
                    model, checkpoint, model_args, args.y_mode)
                model.eval()
                tokens = ckpt_mod.tokens_at_step(step, model_args, None)
                level = str(model_args.level)
                use_triton = int(getattr(model_args, "use_triton", 0) or 0)
                nparams = model.get_num_params()
                log(
                    f"loaded {label} step={step} tokens={tokens:,} level={level} "
                    f"use_triton={use_triton} sf_y_swap={swapped} params={nparams:,}"
                )
                if level in ("E97", "97") and use_triton != 1:
                    raise RuntimeError(
                        f"FUSED-GUARD: emender {ckpt_path} loaded with use_triton={use_triton}"
                        " -- eager forbidden (NON-NEGOTIABLE #1)")
                # REAL fused-guard: verify the recurrence kernel actually fires at
                # eval (config use_triton=1 is necessary but NOT sufficient -- the
                # fused path was gated on self.training and silently fell back to
                # eager under model.eval()). Asserts on instrumented kernel calls.
                guard = assert_fused_recurrence(model, device)
                log(f"  [fused-guard] {label} step={step}: {guard}")

                enc = tiktoken.get_encoding(model_args.tokenizer) if getattr(
                    model_args, "tokenizer", None) else None

                for name, items in pending:
                    score_mode = racer.choose_score_mode("auto", cfg)
                    summary, rows = score_one(
                        model, items, enc, str(device),
                        args.batch_size, score_mode, args.primary_score)
                    prim = summary[args.primary_score]
                    log(
                        f"  {label}/{name} step={step}: "
                        f"acc={prim['correct']}/{prim['n']} "
                        f"({prim['accuracy']:.4f}) mode={score_mode}")
                    # overall row + per-category rows
                    append_row(args.out_csv, {
                        "model": label, "panel": name, "level": level,
                        "step": step, "tokens": tokens, "params": nparams,
                        "sf_y_swap": int(swapped), "score_mode": score_mode,
                        "category": "__overall__", "n": prim["n"],
                        "correct": prim["correct"],
                        "accuracy": f"{prim['accuracy']:.6f}",
                    })
                    for cat, rec in prim["category_summary"].items():
                        append_row(args.out_csv, {
                            "model": label, "panel": name, "level": level,
                            "step": step, "tokens": tokens, "params": nparams,
                            "sf_y_swap": int(swapped), "score_mode": score_mode,
                            "category": cat, "n": rec["n"],
                            "correct": rec["correct"],
                            "accuracy": f"{rec['accuracy']:.6f}",
                        })
                    # per-item correctness for CIs
                    item_recs = []
                    for r in rows:
                        pred = r["preds"][args.primary_score]
                        item_recs.append({
                            "model": label, "panel": name, "step": step,
                            "tokens": tokens, "category": r["category"],
                            "name": r["name"], "answer": r["answer"],
                            "pred": pred, "correct": int(pred == r["answer"]),
                        })
                    append_items(args.out_items, item_recs)
                    already.add((label, name, step))

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                if not args.keep_going:
                    raise
                log(f"ERROR scoring {ckpt_path}: {exc}")
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
