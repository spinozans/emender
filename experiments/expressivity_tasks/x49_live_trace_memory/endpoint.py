"""Durable analysis weights and same-weight within-window scoring. No corpus or split loader."""

from __future__ import annotations
import hashlib
import json
import os
from time import perf_counter
import torch
from torch.nn import functional as F
from ..x48_muon_continuation.data import check, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from .literal import HORIZONS
from .trace import TraceHead
from .write_head import WriteHead


def state_digest(obj):
    """Content digest (not a torch archive digest): bind the entire nested logical state incl. moments."""
    h = hashlib.sha256()

    def data(b):
        v = memoryview(b).cast("B")
        h.update(len(v).to_bytes(8, "big"))
        h.update(v)

    def label(v):
        data(json.dumps(v, allow_nan=False, separators=(",", ":")).encode())

    def visit(x):
        if torch.is_tensor(x):
            label(["tensor", str(x.dtype), list(x.shape)])
            data(
                x.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy()
            )
        elif isinstance(x, dict):
            label(["dict", len(x)])
            for k in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                visit(k)
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            label([type(x).__name__, len(x)])
            for v in x:
                visit(v)
        else:
            label([type(x).__name__, x])

    visit(obj)
    return h.hexdigest()


def milestone(folder, binding, index, host, head):
    """Never replace a prior weights-only milestone. Recovery with moments uses a separate file."""
    path = folder / f"weights-{index:04d}.pt"
    temp = path.with_suffix(".pt.tmp")
    note = path.with_suffix(".json")
    check(not path.exists() and not note.exists(), "analysis milestone already exists")
    with temp.open("xb") as f:
        os.fchmod(f.fileno(), 0o600)
        torch.save(
            dict(
                binding=binding,
                step=index,
                host=host.state_dict(),
                head=head.state_dict(),
            ),
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    temp.rename(path)
    flush_dir(folder)
    r = dict(
        file=path.name,
        bytes=path.stat().st_size,
        step=index,
        binding=binding,
        sha256=sha(path),
    )
    write_json(note, r)
    flush_dir(folder)
    return r


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def short_cells(arm, full):
    check(
        arm in ("B", "P", "L") and type(full) is bool,
        "unknown short predictor/endpoint",
    )
    if not full:
        return [(4096, "none")]
    return (
        [(h, "none") for h in HORIZONS]
        + [(4096, "literal")]
        + ([(4096, "inner")] if arm == "L" else [])
    )


@torch.inference_mode()
def score_window(
    host,
    head,
    word,
    *,
    arm,
    seed,
    document,
    width=256,
    stratify=False,
    short_full_dose=None,
):
    """One source, frozen weights. Full prefill ONCE. Common visit=0, independent memory horizons.

    Returns per-document aggregate sums/work/timing, not activations, targets or scratch records.
    This kernel is not authority to open evaluation text. No host state is returned or kept.
    """
    short = short_full_dose is not None
    check(
        type(word) is bytes
        and (2 <= len(word) <= 4096 if short else len(word) == 4097)
        and (arm in ("B", "P", "L") if short else arm in ("A", "B", "F", "C"))
        and isinstance(head, WriteHead) == (arm in ("P", "L"))
        and (not isinstance(head, WriteHead) or head.arm == arm)
        and not host.training
        and not head.training
        and isinstance(head, TraceHead) == (arm in ("F", "C"))
        and (not isinstance(head, TraceHead) or head.learned == (arm == "C")),
        "one full window, frozen correctly bound eval-mode predictor required",
    )
    dev = next(host.parameters()).device
    if dev.type == "cuda":
        from ..x47_xsa_attention.workbench_run import check_gpu_owner

        check_gpu_owner()
    sync(dev)
    t0 = perf_counter()
    input_bytes = word[:-1]
    ids = torch.tensor([list(input_bytes)], dtype=torch.long, device=dev)
    p = host(ids).to("cpu")
    sync(dev)
    check(
        p.shape == (1, len(input_bytes), width + 256)
        and p.dtype == torch.float32
        and p.isfinite().all(),
        "invalid eval host packet",
    )
    h, z = p[0].split((width, 256), dim=-1)
    prefill = perf_counter() - t0
    cells = [(v, "none") for v in HORIZONS]
    if arm != "A":
        cells.append((4096, "literal"))
    if arm in ("F", "C"):
        cells.append((4096, "trace"))
    if short:
        cells = short_cells(arm, short_full_dose)
    rows = []
    for horizon, sham in cells:
        tic = perf_counter()
        work = {}
        ages = {} if stratify else None
        inspect = horizon == 4096 and arm != "A" and short_full_dose is not False
        if isinstance(head, WriteHead):
            lp = head(
                input_bytes,
                h,
                z,
                mode=arm,
                horizon=horizon,
                inner_sham=sham == "inner",
                outer_sham=sham == "literal",
                inspect_rotation=inspect,
                work=work,
                age_out=ages,
            )
        elif isinstance(head, TraceHead):
            lp = head(
                input_bytes,
                h,
                z,
                seed=seed,
                document=document,
                visit=0,
                horizon=horizon,
                wrong_binding=sham == "trace",
                literal_sham=sham == "literal",
                inspect_rotation=inspect,
                work=work,
                age_out=ages,
            ).lp
        else:
            lp = head(
                input_bytes,
                h,
                z,
                mode="rotated" if sham == "literal" else arm,
                horizon=horizon,
                inspect_rotation=inspect,
                work=work,
                age_out=ages,
            )
        # Expose the shift ONLY to the scoring seam; every real byte, no mask on misses.
        ys = torch.tensor(list(word[1:]), dtype=torch.long)
        losses = F.nll_loss(lp, ys, reduction="none").double()
        total = losses.sum().item()
        strata = {}
        for source, age in (ages or {}).items():
            groups = {
                name: dict(targets=int(mask.sum()), nll_sum=losses[mask].sum().item())
                for name, mask in [
                    ("miss", age == 0),
                    ("16_to_128", (age >= 16) & (age <= 128)),
                    ("129_to_1024", (age > 128) & (age <= 1024)),
                    ("over_1024", age > 1024),
                ]
            }
            check(
                sum(x["targets"] for x in groups.values()) == len(ys)
                and groups["miss"]["targets"] == work[source + "_misses"]
                and abs(sum(x["nll_sum"] for x in groups.values()) - total) < 1e-8,
                "age strata do not partition all targets",
            )
            strata[source] = groups
        check(
            lp.shape == (len(ys), 256)
            and 0 <= total < float("inf")
            and work["literal_hits"] + work["literal_misses"] == len(ys),
            "wrong eval denominator",
        )
        rows.append(
            dict(
                horizon=horizon,
                sham=sham,
                targets=len(ys),
                nll_sum=total,
                work=work,
                strata=strata,
                head_score_s=perf_counter() - tic,
            )
        )
    return dict(
        host_prefills=1,
        host_tokens=len(input_bytes),
        host_prefill_s=prefill,
        cells=rows,
    )
