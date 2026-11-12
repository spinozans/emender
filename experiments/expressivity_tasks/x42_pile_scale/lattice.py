"""X43 150M lattice runner: registered dense arms on the registered flagship stream.

This is a *measurement* harness, distinct from the search harness in
``byte_velocity.py``. It deliberately owns its own bindings so that a run can never
silently measure the wrong corpus:

* the stream is the flagship manifest (documents fully beyond byte 100,000,000,000),
  hash-verified against ``receipts/flagship_stream/registration.json``;
* the budget, arms and eval protocol come from PREREGISTRATION_150M_LATTICE.md;
* every result receipt binds the lattice registration, the stream identity and the
  nominated search policy it was seeded from.

Arms 5-7 (the two consultation arms and the append-only control that twins arm 6)
are gated on the live consult path and are NOT runnable here; asking for them fails
closed rather than approximating them.

Usage
    python -m experiments.expressivity_tasks.x42_pile_scale.lattice train \
        --arm lattice_emender_bounded --seconds 2700 --seed 3401 --output DIR
    python -m experiments.expressivity_tasks.x42_pile_scale.lattice gate   # CPU gates
    python -m experiments.expressivity_tasks.x42_pile_scale.lattice gate --gpu  # + GPU gates
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .byte_velocity import CAPACITY, grouped, model_for
from .velocity_candidate import Policy
from .velocity_baseline_search import SearchPolicy
from .velocity_state import atomic_json
from .velocity_train import configure_production

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
X43 = REPO / "experiments/expressivity_tasks/x43_fork_resolved"
FLAGSHIP = X43 / "receipts/flagship_stream"
LATTICE_REG = X43 / "PREREGISTRATION_150M_LATTICE.md"
FLAGSHIP_REG = FLAGSHIP / "registration.json"
FLAGSHIP_MANIFEST = FLAGSHIP / "documents.jsonl.gz"
NOMINATIONS = ROOT / "receipts/byte_velocity_nominations"
PYTHON = "/home/erikg/emender/.venv/bin/python"

CONTEXT = 2048
BOUNDARY_BYTES = 100_000_000_000
REGISTERED_SPLITS = ("train", "calibration", "final_test")
# "Unbounded" is a finite cap that provably never binds, receipted as
# finite_cap_never_bound (the C14-B convention, budget=257 there). Here the store resets
# every window and at most one slot can be appended per token, so occupancy <= CONTEXT;
# a cap of CONTEXT+1 cannot bind. The native ceiling (262144) is rejected by the Triton
# kernel, so the provable bound is also the safe one.
UNBOUNDED_BUDGET = CONTEXT + 1

# Dense wave (arms 1-4 of the lattice registration). Consulted arms 5-7 are gated.
ARMS = {
    "lattice_transformer": dict(base="transformer", nomination="transformer", unbounded=False),
    "lattice_gdn2": dict(base="gdn2", nomination="gdn2", unbounded=False),
    "lattice_emender_bounded": dict(base="pure", nomination="pure", unbounded=False),
    "lattice_emender_unbounded": dict(base="pure", nomination="pure", unbounded=True),
}
GATED_ARMS = ("lattice_transformer_xsa", "lattice_emender_xsa", "lattice_emender_xsa_append_only")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_python():
    if os.path.realpath(os.sys.executable) != os.path.realpath(PYTHON):
        raise RuntimeError(f"approved Python required (got {os.sys.executable})")


def flagship_identity() -> dict:
    """Verify the registered flagship stream before training on it."""
    reg = json.loads(FLAGSHIP_REG.read_text())
    source = Path(reg["source"]["path"])
    st = source.stat()
    if (st.st_size, st.st_ino, st.st_mtime_ns) != (
        reg["source"]["size"], reg["source"]["inode"], reg["source"]["mtime_ns"]):
        raise ValueError("flagship source identity changed: refusing to train")
    return dict(
        stream="flagship_beyond_100gb",
        source=str(source), source_size=st.st_size,
        manifest=str(FLAGSHIP_MANIFEST), manifest_sha256=digest(FLAGSHIP_MANIFEST),
        registration=str(FLAGSHIP_REG), registration_sha256=digest(FLAGSHIP_REG),
        boundary_bytes=BOUNDARY_BYTES,
        lattice_registration=str(LATTICE_REG), lattice_sha256=digest(LATTICE_REG),
    )


def nomination(arm: str) -> dict:
    path = NOMINATIONS / f"{arm}.json"
    if not path.exists():
        raise ValueError(f"missing nomination receipt: {path}")
    rec = json.loads(path.read_text())
    best = rec["nomination"]
    return dict(arm=arm, receipt=str(path), receipt_sha256=digest(path),
                generation=best["generation"], candidate=best["candidate"],
                policy=best["policy"], search_score=best["score"])


class Stream:
    """Document-bounded byte windows from an explicit manifest (plain or gzipped JSONL).

    Every window is a slice of exactly one document; a window never crosses a document
    start. Each document slice is SHA-256 verified at consumption.  The manifest is the
    sole authority for boundaries: no byte value is ever scanned for.
    """

    def __init__(self, source, manifest, context=CONTEXT, verify_document=True):
        self.source = Path(source)
        self.manifest = Path(manifest)
        self.context = context
        self.verify_document = verify_document

    def _lines(self):
        opener = gzip.open if self.manifest.suffix == ".gz" else open
        with opener(self.manifest, "rt") as f:
            for line in f:
                yield json.loads(line)

    def documents(self, split):
        for d in self._lines():
            if d.get("split") != split:
                continue
            doc = d.get("document", d.get("source_index"))
            yield dict(doc=doc, offset=d["byte_offset"], length=d["byte_length"],
                       sha256=d["sha256"], split=split)

    def documents_iter(self):
        for d in self._lines():
            yield dict(doc=d.get("document", d.get("source_index")),
                       offset=d["byte_offset"], length=d["byte_length"],
                       sha256=d["sha256"], split=d.get("split"))

    def windows(self, split, context=None):
        context = self.context if context is None else context
        with self.source.open("rb") as raw:
            for d in self.documents(split):
                raw.seek(d["offset"])
                text = raw.read(d["length"])
                if len(text) != d["length"]:
                    raise ValueError(f"short read at document {d['doc']}")
                if self.verify_document and hashlib.sha256(text).hexdigest() != d["sha256"]:
                    raise ValueError(f"document hash mismatch: {d['doc']} — wrong stream?")
                for start in range(0, len(text) - 1, context):
                    piece = np.frombuffer(text[start:start + context + 1], dtype=np.uint8)
                    piece = piece.astype(np.int64)
                    if start + len(piece) - 1 > len(text):
                        raise ValueError("window crossed document end")
                    yield d["doc"], start, piece[:-1], piece[1:]

    def batches(self, split, context=None, batch=64):
        rows = []
        for row in self.windows(split, context):
            rows.append(row)
            if len(rows) == batch:
                yield rows
                rows = []
        if rows:
            yield rows


def assert_flagship_stream(stream: "Stream") -> dict:
    """Structural fail-closed check on the stream a lattice arm is about to consume.

    A hash check is not enough: the C14 in-region manifest also hashes cleanly against
    pile.txt, so a wrong stream is *valid* and would train happily.  The lattice stream
    is accepted only if (a) every document lies fully beyond BOUNDARY_BYTES and (b) the
    split names are the registered three.
    """
    seen = {}
    below = []
    for d in stream.documents_iter():
        seen[d["split"]] = seen.get(d["split"], 0) + 1
        if d["offset"] < BOUNDARY_BYTES:
            below.append(d["doc"])
            if len(below) > 4:
                break
    if below:
        raise ValueError(
            f"wrong stream: {len(below)}+ documents start below the {BOUNDARY_BYTES} "
            "boundary — the lattice trains only on the registered beyond-boundary stream")
    bad = set(seen) - set(REGISTERED_SPLITS)
    if bad:
        raise ValueError(f"wrong stream: unregistered splits {sorted(bad)}")
    missing = set(REGISTERED_SPLITS) - set(seen)
    if missing:
        raise ValueError(f"wrong stream: missing splits {sorted(missing)}")
    return dict(documents=seen)


def build_policy(arm: str) -> tuple[str, Policy, dict]:
    spec = ARMS.get(arm)
    if spec is None:
        raise ValueError(
            f"arm {arm!r} is not runnable: dense wave is {sorted(ARMS)}; "
            f"gated on the live consult path: {GATED_ARMS}")
    nom = nomination(spec["nomination"])
    cls = Policy if spec["base"] == "pure" else SearchPolicy
    policy = cls(**nom["policy"])
    if spec["unbounded"]:
        if spec["base"] != "pure":
            raise ValueError("unbounded is defined only for the emender arm")
        # cap non-binding: same code path, budget above what any window can populate
        policy = replace(policy, budget_min=UNBOUNDED_BUDGET, budget_ratio=1.0)
    return spec["base"], policy, nom


def evaluate(model, stream, arm, split, windows=24):
    model.eval()
    total = 0.0
    targets = 0
    used = []
    with torch.no_grad():
        for i, row in enumerate(stream.windows(split)):
            if i >= windows:
                break
            if len(row[2]) != CONTEXT:
                continue  # full windows only, identical rule to the search protocol
            x, y, _ = next(grouped([row], 1))
            total += float(model.loss(x, y)) * y.numel()
            targets += int(y.numel())
            used.append([row[0], row[1]])
    if not targets:
        raise ValueError(f"no full {CONTEXT}-byte windows in split {split}")
    return dict(score=total / targets, nll_heldout=total / targets,
                eval_split=split, eval_windows=len(used), eval_targets=targets,
                units="nats/byte")


def train(arm, seconds, seed, output, eval_split="calibration", device=None, quiet=False):
    check_python()
    identity = flagship_identity()
    base, policy, nom = build_policy(arm)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    card = device or os.environ.get("GPU_LEASE_HELD")
    if card is None:
        raise ValueError("no GPU lease: set GPU_LEASE_HELD (or --device) before training")
    if visible is None:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be set before launch (a recorded card label is not "
            "a lease: model.cuda() would otherwise land on physical device 0)")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: recorded card {card} vs CUDA_VISIBLE_DEVICES {visible}")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    configure_production()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    model = model_for(base, policy).cuda().train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=policy.lr,
        betas=(.9, .95 if base == "pure" else policy.beta2),
        weight_decay=.1 if base == "pure" else policy.weight_decay, fused=True)
    stream = Stream(identity["source"], identity["manifest"])
    assert_flagship_stream(stream)
    atomic_json(out / "start.json", dict(
        arm=arm, base_arm=base, policy=asdict(policy), seed=seed,
        nomination={k: nom[k] for k in ("arm", "receipt", "receipt_sha256", "generation",
                                        "candidate", "search_score")},
        identity=identity, physical_gpu=card, pid=os.getpid(),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad)),
        budgets=list(_budgets(policy)),
        context=CONTEXT, units="nats/byte"))
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = targets = 0
    with (out / "training.jsonl").open("w", buffering=1) as log:
        while elapsed < seconds:
            torch.cuda.synchronize()
            start = time.monotonic()
            try:
                rows = next(batches)
            except StopIteration:
                raise RuntimeError("train split exhausted before the budget: refusing to wrap")
            n = sum(len(r[3]) for r in rows)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = policy.lr * min(1., (step + 1) / max(1, policy.warmup_steps))
            loss_sum = 0.0
            for x, y, valid in grouped(rows, 8 if base == "pure" else policy.microbatch):
                loss = model.loss(x, y)
                if base == "pure":
                    loss = loss * (y.numel() / valid) + policy.gate_cost_lambda * model.gate_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                (loss * (valid / n)).backward()
                loss_sum += float(loss.detach()) * valid / n
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - start
            step += 1
            targets += n
            log.write(json.dumps(dict(step=step, active_seconds=elapsed, bytes=targets,
                                      loss=loss_sum, time_ns=time.time_ns())) + "\n")
            if not quiet and step % 50 == 0:
                print(f"  {arm} step {step} {elapsed:.0f}s {targets/1e6:.1f}MB "
                      f"loss {loss_sum:.4f}", flush=True)
    os.environ['X40_BATCHED_READS'] = '0'  # evaluation path is recorded-read only
    result = evaluate(model, stream, base, eval_split)
    result.update(status="completed", score_eligible=True, arm=arm, base_arm=base,
                  train_seconds=elapsed, steps=step, bytes=targets,
                  seed=seed, units="nats/byte", identity=identity,
                  policy=asdict(policy), budgets=list(_budgets(policy)))
    atomic_json(out / "result.json", result)
    return result


# ---------------------------------------------------------------- gates

def gate_stream_binding():
    """A valid-but-wrong manifest must fail closed, not train."""
    identity = flagship_identity()
    wrong = Path("/home/erikg/c14_dense_bytes/pile_offsets.jsonl")  # C14 in-region stream
    if not wrong.exists():
        return dict(gate="stream_binding", status="skipped", reason="no alternate manifest")
    try:
        assert_flagship_stream(Stream(identity["source"], wrong))
    except ValueError as e:
        return dict(gate="stream_binding", status="pass", detail=str(e))
    raise AssertionError("alternate stream did not fail closed")


def gate_boundary():
    """Every window is inside one document and starts at a document-relative offset."""
    identity = flagship_identity()
    s = Stream(identity["source"], identity["manifest"])
    docs = {d["doc"]: d for d in s.documents("train")}
    seen = 0
    max_start = 0
    for doc, start, x, y in s.windows("train"):
        d = docs[doc]
        if start + len(x) + 1 > d["length"]:
            raise AssertionError(f"window crossed document end: {doc} {start}")
        if start % CONTEXT:
            raise AssertionError("unaligned window start")
        if start + len(x) + 1 > BOUNDARY_BYTES and d["offset"] < BOUNDARY_BYTES:
            raise AssertionError("window in the beyond-boundary stream starts below the boundary")
        seen += 1
        max_start = max(max_start, start)
        if seen >= 20000:
            break
    return dict(gate="boundary", status="pass", windows_checked=seen, max_start=max_start)


def gate_splits():
    """No document appears in two splits (by id and by content hash)."""
    identity = flagship_identity()
    s = Stream(identity["source"], identity["manifest"])
    ids, hashes = {}, {}
    for split in ("train", "calibration", "final_test"):
        for d in s.documents(split):
            if d["doc"] in ids and ids[d["doc"]] != split:
                raise AssertionError(f"document {d['doc']} in {ids[d['doc']]} and {split}")
            ids[d["doc"]] = split
            if d["sha256"] in hashes and hashes[d["sha256"]] != split:
                raise AssertionError(f"content hash shared across splits: {d['sha256']}")
            hashes[d["sha256"]] = split
    counts = {s: sum(1 for v in ids.values() if v == s) for s in set(ids.values())}
    return dict(gate="splits", status="pass", documents=counts)


def _budgets(policy):
    b = getattr(policy, "budgets", None)
    if b is None:
        return tuple()
    return b() if callable(b) else tuple(b)


def gate_arms():
    """The arms must be genuinely distinct configurations."""
    seen = {}
    for arm in ARMS:
        base, policy, nom = build_policy(arm)
        d = asdict(policy)
        key = (base, d.get("budget_min"), d.get("budget_ratio"), d.get("lr"))
        if key in seen:
            raise AssertionError(f"{arm} is not distinct from {seen[key]}")
        seen[key] = arm
        if base == "pure":
            if arm.endswith("unbounded"):
                if max(_budgets(policy)) < UNBOUNDED_BUDGET:
                    raise AssertionError("unbounded arm still has a binding cap")
            elif max(_budgets(policy)) >= UNBOUNDED_BUDGET:
                raise AssertionError(f"{arm} has a non-binding cap: mislabelled")
    for arm in GATED_ARMS:
        try:
            build_policy(arm)
        except ValueError as e:
            if "not runnable" not in str(e):
                raise
        else:
            raise AssertionError(f"gated arm {arm} was buildable")
    return dict(gate="arms", status="pass", arms={k: v["base"] for k, v in ARMS.items()})


def gate_repro(device="5", seconds=90):
    """Two short runs of the same config must produce identical losses."""
    runs = []
    for tag in ("r1", "r2"):
        out = Path(f"/tmp/lattice-gate-repro-{tag}-{os.getpid()}")
        if out.exists():
            raise RuntimeError(f"stale gate dir {out}")
        r = train("lattice_emender_bounded", seconds, 3401, out, device=device, quiet=True)
        losses = [json.loads(l)["loss"] for l in (out / "training.jsonl").read_text().splitlines()]
        runs.append((losses, r["score"]))
    n = min(len(runs[0][0]), len(runs[1][0]), 20)
    if runs[0][0][:n] != runs[1][0][:n]:
        raise AssertionError(f"non-reproducible: {runs[0][0][:3]} vs {runs[1][0][:3]}")
    return dict(gate="repro", status="pass", steps_compared=n, eval_scores=[runs[0][1], runs[1][1]])


def main():
    # production allocator settings must be in place before CUDA initializes
    for flag in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        os.environ.setdefault(flag, "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True)
    t.add_argument("--seconds", type=float, default=2700.)
    t.add_argument("--seed", type=int, default=3401)
    t.add_argument("--output", required=True)
    t.add_argument("--eval-split", default="calibration")
    t.add_argument("--device", default=None)
    g = sub.add_parser("gate")
    g.add_argument("--gpu", action="store_true")
    g.add_argument("--device", default="5")
    a = ap.parse_args()
    if a.cmd == "train":
        r = train(a.arm, a.seconds, a.seed, a.output, a.eval_split, a.device)
        print(json.dumps({k: r[k] for k in ("arm", "score", "eval_split", "eval_windows",
                                            "steps", "bytes", "train_seconds")}, indent=2))
        return
    results = [gate_stream_binding(), gate_splits(), gate_arms(), gate_boundary()]
    if a.gpu:
        results.append(gate_repro(a.device))
    for r in results:
        print(json.dumps(r))
    bad = [r for r in results if r["status"] not in ("pass", "skipped")]
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
