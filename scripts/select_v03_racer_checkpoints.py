#!/usr/bin/env python3
"""Select the V3 racer checkpoints that match the committed V3 paper endpoints.

Read-only selection + evidence helper for `v3-validate-current`. For each
architecture it:

  1. Re-runs the *canonical* Figure 2 smoothing (`paper/results/figure_2/smooth.py`
     MODELS config + `trailing_average`, the same method that produced the paper
     numbers) against the current live racer logs.
  2. Picks the retained on-disk checkpoint `.pt` nearest the V3 paper AS_OF step
     (the committed `main.typ` / `AS_OF.md` endpoint, NOT the latest training
     step), and records the step delta from both the committed-paper endpoint and
     the (now-stale) ticket AS_OF endpoint.
  3. Records the `trail_100k` smoothed loss/BPB at the selected checkpoint step,
     plus path / mtime / size / SHA256 / raw filename loss.

It performs NO conversion, NO Hugging Face calls, and writes only a small JSON
evidence file under the requested workdir. It does not stage anything in git.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SMOOTH_PY = ROOT / "paper/results/figure_2/smooth.py"

# Pinned tokenizer constant (authoritative; same as V3_NUMBERS.md / AS_OF.md).
TOK_JSON = ROOT / "scripts/estimate_tokenizer_bytes_per_token.json"
with TOK_JSON.open() as _fh:
    _TOK = json.load(_fh)
BPB_PER_NAT = _TOK["bits_per_byte_per_nat_per_token"]  # 0.3681635882200934
BYTES_PER_TOKEN = _TOK["mean_bytes_per_token"]  # 3.918625

# Committed V3 paper endpoints (main.typ §5 / AS_OF.md, 2026-05-31T13:49:33Z snapshot).
PAPER_ENDPOINT = {
    "e88": {"step": 1_523_250, "trail_100k_loss": 2.644925, "bpb": 0.973765},
    "gdn": {"step": 1_999_300, "trail_100k_loss": 2.653617, "bpb": 0.976965},
    "m2rnn": {"step": 1_466_400, "trail_100k_loss": 2.661439, "bpb": 0.979845},
}

# Ticket AS_OF steps (PRE-recompute draft; superseded by v3-data-recompute).
TICKET_AS_OF = {"e88": 1_405_450, "gdn": 1_847_050, "m2rnn": 1_343_050}

# Map our keys to smooth.py MODELS keys and on-disk checkpoint directories.
SMOOTH_KEY = {"e88": "E88_NDM", "gdn": "FLA_GDN", "m2rnn": "M2RNN_CMA"}
CHECKPOINT_DIR = {
    "e88": Path(
        "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/"
        "levelE88_1270M_20260511_233832"
    ),
    "gdn": Path(
        "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/"
        "levelfla-gdn_1270M_20260511_233832"
    ),
    "m2rnn": Path(
        "/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/"
        "levelm2rnn_1270M_20260511_175023"
    ),
}

CHECKPOINT_RE = re.compile(r"checkpoint_step_(?P<step>\d+)_loss_(?P<loss>\d+(?:\.\d+)?)\.pt$")


def load_smooth_module():
    spec = importlib.util.spec_from_file_location("figure2_smooth", SMOOTH_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc_mtime(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(128 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_checkpoints(key: str) -> list[dict[str, Any]]:
    out = []
    for ckpt in sorted(CHECKPOINT_DIR[key].glob("checkpoint_step_*_loss_*.pt")):
        m = CHECKPOINT_RE.match(ckpt.name)
        if not m:
            continue
        out.append(
            {
                "path": str(ckpt),
                "step": int(m.group("step")),
                "raw_filename_loss": float(m.group("loss")),
            }
        )
    return out


def compute_trail_100k(smooth_mod, key: str) -> dict[int, float]:
    """Return {step: trail_100k_loss} using the canonical smooth.py windowing."""
    cfg = smooth_mod.MODELS[SMOOTH_KEY[key]]
    log_paths = [entry[0] for entry in cfg["logs"]]
    rows = smooth_mod.merge_logs(
        log_paths,
        cfg["params"],
        cfg["batch_size"],
        cfg["chunk_size"],
        cfg["bytes_per_token"],
    )
    steps = np.array([r["step"] for r in rows])
    losses = np.array([r["loss"] for r in rows])
    log_every = int(np.median(np.diff(steps))) if len(steps) > 1 else 50
    idx_win = max(1, 100_000 // log_every)
    trailed = smooth_mod.trailing_average(losses, idx_win)
    return {int(s): float(t) for s, t in zip(steps, trailed)}, log_every, len(rows)


def select(args) -> dict[str, Any]:
    smooth_mod = load_smooth_module()
    result: dict[str, Any] = {
        "method": (
            "Re-ran canonical paper/results/figure_2/smooth.py MODELS + "
            "trailing_average against current live logs; selected the retained "
            ".pt nearest the committed V3 paper AS_OF step (trail_100k convention)."
        ),
        "bpb_per_nat": BPB_PER_NAT,
        "bytes_per_token": BYTES_PER_TOKEN,
        "paper_endpoint": PAPER_ENDPOINT,
        "ticket_as_of": TICKET_AS_OF,
        "models": {},
    }
    for key in args.models:
        trail_map, log_every, n_rows = compute_trail_100k(smooth_mod, key)
        ckpts = list_checkpoints(key)
        avail_steps = [c["step"] for c in ckpts]
        paper_step = PAPER_ENDPOINT[key]["step"]
        # absolute nearest checkpoint to the committed paper endpoint step
        chosen = min(ckpts, key=lambda c: abs(c["step"] - paper_step))
        cpath = Path(chosen["path"])
        # trail_100k at the chosen checkpoint step (exact row; steps are /50)
        t100k = trail_map.get(chosen["step"])
        result["models"][key] = {
            "checkpoint_dir": str(CHECKPOINT_DIR[key]),
            "retained_steps": avail_steps,
            "log_every": log_every,
            "log_rows": n_rows,
            "ticket_as_of_step": TICKET_AS_OF[key],
            "ticket_as_of_available_on_disk": TICKET_AS_OF[key] in avail_steps,
            "paper_endpoint_step": paper_step,
            "paper_endpoint_trail_100k_loss": PAPER_ENDPOINT[key]["trail_100k_loss"],
            "paper_endpoint_bpb": PAPER_ENDPOINT[key]["bpb"],
            "selected": {
                "path": str(cpath),
                "step": chosen["step"],
                "delta_vs_paper_step": chosen["step"] - paper_step,
                "delta_vs_ticket_as_of": chosen["step"] - TICKET_AS_OF[key],
                "raw_filename_loss": chosen["raw_filename_loss"],
                "raw_filename_bpb": chosen["raw_filename_loss"] * BPB_PER_NAT,
                "trail_100k_loss_at_step": t100k,
                "trail_100k_bpb_at_step": (t100k * BPB_PER_NAT) if t100k else None,
                "trail_100k_bpb_minus_paper_bpb": (
                    (t100k * BPB_PER_NAT) - PAPER_ENDPOINT[key]["bpb"]
                    if t100k
                    else None
                ),
                "mtime_utc": utc_mtime(cpath),
                "size_bytes": cpath.stat().st_size,
                "sha256": sha256_file(cpath) if args.hash else None,
            },
        }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="*", choices=sorted(SMOOTH_KEY), default=sorted(SMOOTH_KEY))
    p.add_argument("--out", type=Path, default=Path("/tmp/v3-racer-selection.json"))
    p.add_argument("--hash", action="store_true", help="compute SHA256 (slow; reads full .pt)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = select(args)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
