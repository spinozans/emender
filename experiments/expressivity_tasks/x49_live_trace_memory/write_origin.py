"""Load the actual successful v2 B reader/host for each seed, not X48 or a learned-key endpoint."""

from __future__ import annotations

from pathlib import Path
import torch
from ..x48_muon_continuation.data import check, load, sha
from .coupled import cpu_origin
from .panel_spec import ROOT as OLD, binding as v2_binding, SEEDS
from .panel_report import TRAIN_REF
from .write_head import WriteHead

RESULT = "b5242f87e1f3ba03402fd407153c2f4d77f95a30b4182aebd079d17f87b7e121"


def bound_b(seed):
    p = Path(__file__).with_name("receipts") / "panel_result.json"
    check(
        type(seed) is int and seed in SEEDS and sha(p) == RESULT,
        "wrong B result identity",
    )
    return load(p)["training"][f"B-{seed}"]["weights_sha"]


def origin(seed, arm):
    check(arm in ("B", "P", "L"), "not a registered W arm")
    d = OLD / f"B-{seed}"
    p = d / "weights-1024.pt"
    h = bound_b(seed)
    b = v2_binding(TRAIN_REF, "B", seed)
    check(
        load(p.with_suffix(".json"))
        == dict(bytes=p.stat().st_size, file=p.name, step=1024, sha256=h, binding=b)
        and sha(p) == h,
        "not the bound full-dose B endpoint",
    )
    m, c = cpu_origin(seed, "B")
    ck = torch.load(p, weights_only=True, mmap=True, map_location="cpu")
    check(
        set(ck) == {"step", "binding", "host", "head"}
        and ck["step"] == 1024
        and ck["binding"] == b,
        "unbound B state",
    )
    m.load_state_dict(ck["host"], strict=True)
    c.load_state_dict(ck["head"], strict=True)
    return m.train(), (c if arm == "B" else WriteHead(c, arm=arm, seed=seed)).train()
