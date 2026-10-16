"""Evaluate retained checkpoints at the union of headline stopping steps.

No training is performed. Every selection_frozen receipt must already exist.
A checkpoint beyond an arm's stop is unavailable, never silently extrapolated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.expressivity_tasks.run_invertible_stage2 import (
    digest, evaluate, remove_state_tanh, write_json,
)


def main():
    import numpy as np
    import torch
    from scripts.expressivity_1p3b import load_model
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = args.root / args.run
    cfg = json.loads((run / "config.json").read_text())
    result = json.loads((run / "result.json").read_text())
    stops = set()
    for other in args.root.glob("headline_*"):
        frozen = json.loads((other / "selection_frozen.json").read_text())
        if not frozen["frozen"]:
            raise RuntimeError(f"selection not frozen: {other}")
        stops.add(frozen["stop_step"])
    checkpoints = json.loads((run / "checkpoints.json").read_text())
    matched = json.loads((run / "matched_test.json").read_text())
    additional = {}
    data = np.load(run / "frozen_pools.npz")
    pools = {name: tuple(data[f"{name}_{i}"] for i in range(3))
             for name in ("test_1x", "test_4x", "test_8x")}
    needed = [s for s in sorted(stops) if s <= result["stop_step"] and str(s) not in matched]
    available = [s for s in needed if Path(checkpoints[str(s)]["path"]).exists()]
    if available:
        model = load_model("gdn2" if cfg["arm"] == "gdn2" else "e97", torch.device("cuda"))
        if cfg["arm"] == "e97_identity":
            remove_state_tanh(model)
        for step in available:
            cp = checkpoints[str(step)]
            if digest(cp["path"]) != cp["sha256"]:
                raise RuntimeError("matched checkpoint hash mismatch")
            raw = torch.load(cp["path"], map_location="cpu", weights_only=False)
            if raw["completed_updates"] != step:
                raise RuntimeError("checkpoint update count mismatch")
            model.load_state_dict(raw["model"], strict=True)
            del raw
            additional[str(step)] = {name: evaluate(model, pool) for name, pool in pools.items()}
    panels = {}
    for step in sorted(stops):
        if step > result["stop_step"]:
            panels[str(step)] = {"status": "unavailable", "reason": "arm already stopped; no extra training"}
        elif str(step) not in matched and str(step) not in additional:
            panels[str(step)] = {"status": "unavailable", "reason": "checkpoint pruned before stop step became required"}
        else:
            panels[str(step)] = {"status": "evaluated", "checkpoint": checkpoints[str(step)],
                                 "panels": matched.get(str(step), additional.get(str(step)))}
            if panels[str(step)]["panels"] is None:
                raise RuntimeError("required matched-step receipt missing")
    write_json(run / "cross_plateau_test.json", panels)


if __name__ == "__main__":
    main()
