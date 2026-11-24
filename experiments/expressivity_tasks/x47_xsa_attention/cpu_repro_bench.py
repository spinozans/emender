"""T5. Six finite, independent CPU worker processes on previously verified TRAIN features.

Nothing starts or acquires a GPU. Do not reuse or feed a previous worker's snapshot to a learner.
"""

from __future__ import annotations
import argparse, hashlib, os, subprocess, sys
from pathlib import Path
from time import perf_counter_ns as clock, time_ns
import numpy as np
import torch
from ndm.memory_program import PackedMemoryPolicy
from ndm.memory_program.frozen_packet import FrozenPacket
from .frozen_packet_bench import CAP, SEED, canonical, no_card, step
from .memory_program_profile import training_data
from .serving_bench import CK, CK_SHA
from .serving_verify import HERE, load
from .stream_packet_bench import binding
from .workbench_run import dump, sha
from .workbench_witness import digest_state, require

ORDER = (
    "retained-ordinary-A",
    "retained-strict-A",
    "replicated-strict-A",
    "retained-strict-B",
    "replicated-strict-B",
    "retained-ordinary-B",
)
PAIRS = (
    ("retained-strict-A", "retained-strict-B"),
    ("replicated-strict-A", "replicated-strict-B"),
    ("retained-ordinary-A", "retained-ordinary-B"),
    ("retained-ordinary-A", "retained-strict-A"),
)
FWD = (
    "input_sha",
    "byte_vectors",
    "joint_scores",
    "events",
    "per_document_events",
    "dispatch",
    "nll",
    "objective",
    "before",
)
PRIOR = HERE / "receipts/memory-program-t4/attempt1"
MODULE = "experiments.expressivity_tasks.x47_xsa_attention.cpu_repro_bench"


def summary(a, b):
    if (
        a.dtype != np.float32
        or b.dtype != a.dtype
        or a.shape != b.shape
        or a.size == 0
        or not all(np.isfinite(t).all() for t in (a, b))
    ):
        raise ValueError("incompatible or nonfinite snapshot")
    x = a.reshape(-1)
    y = b.reshape(-1)
    mismatch = int(np.count_nonzero(x.view(np.uint32) != y.view(np.uint32)))
    err = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
    mag = float(np.abs(x).max())
    return dict(
        elements=x.size, unequal=mismatch, max_abs=err, ref_absmax=mag, within_t1=err <= 2e-4 + 2e-4 * mag
    )


def fingerprint(v):
    return hashlib.sha256(v.tobytes()).hexdigest()


def worker(root, cell):
    no_card()
    os.umask(0o077)
    start = time_ns()
    size, arm, _ = cell.split("-")
    strict = arm == "strict"
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(strict, warn_only=False)
    torch.manual_seed(SEED)
    require(
        torch.get_num_threads() == 4
        and torch.are_deterministic_algorithms_enabled() == strict
        and not torch.is_deterministic_algorithms_warn_only_enabled(),
        "worker mode",
    )
    run = root / cell
    run.mkdir(mode=0o700, exist_ok=False)
    prod = load(PRIOR / "identity.json")
    meta = load(PRIOR / f"{size}-packet.json")
    ref = load(PRIOR / f"{size}-coupled.json")
    require(sha(CK) == CK_SHA and meta["payload_sha"] == ref["checks"]["input_sha"], "wrong prior authority")
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    p = PackedMemoryPolicy(256).train()
    p.load_state_dict(ck["model"])
    require(digest_state(p) == prod["policy_sha"] == ck["policy_sha"], "initial policy")
    ds = training_data(32)
    rows, ws, man, data = ds
    require(
        dict(data, window129_sha=[hashlib.sha256(w).hexdigest() for w in ws]) == prod["data"]
        and man["mount"] == prod["mount"],
        "changed data",
    )
    b, spec = binding(size, rows, ws, data, prod)
    require(meta["model_spec"] == spec, "changed producer")
    local = Path("output/memory-program-t4/attempt1") / f"{size}.f32"
    begin = clock()
    packet = FrozenPacket.load(local, b, meta["sha256"], byte_limit=CAP)
    load_ns = clock() - begin
    snap = {}

    def capture(group, k, t):
        key = f"{group}:{k}"
        if t.device.type != "cpu" or key in snap:
            raise ValueError("wrong snapshot binding")
        snap[key] = t.detach().contiguous().numpy().copy()

    def prog(phase, k):
        dump(run / "progress.json", dict(phase=phase, scored=k, learner_pid=os.getpid()))

    rep = step(run, "step", p, packet.encoder, ds, prog, coupled=False, capture=capture)
    c = rep["checks"]
    require(
        all(c[k] == ref["checks"][k] for k in FWD)
        and rep["activity"] == ref["activity"]
        and rep["dispatches"] == ref["dispatches"],
        "changed forward workload",
    )
    t = clock()
    path = run / "private-tensors.npz"
    with path.open("xb") as f:
        np.savez(f, **snap)
    after = hashlib.sha256()
    mom = {}
    for k in sorted(c["gradients"]):
        v = snap["param:" + k]
        after.update(f"{k}|torch.float32|{tuple(v.shape)}\0".encode())
        after.update(v.tobytes())
        if c["gradients"][k] is not None:
            mom[k] = {s: fingerprint(snap["adam." + s + ":" + k]) for s in ("step", "exp_avg", "exp_avg_sq")}
    require(after.hexdigest() == c["after"] and canonical(mom) == c["optimizer"], "wrong post-step snapshot")
    digests = {k: dict(sha=fingerprint(v), shape=list(v.shape)) for k, v in snap.items()}
    require(
        all(
            c["gradients"][k] == (digests["grad:" + k]["sha"] if "grad:" + k in digests else None)
            for k in c["gradients"]
        ),
        "changed gradient snapshot",
    )
    no_card()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dump(
        run / "identity.json",
        dict(
            ref=head,
            pid=os.getpid(),
            parent=os.getppid(),
            cell=cell,
            strict=strict,
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            cuda_available=torch.cuda.is_available(),
            cuda_initialized=torch.cuda.is_initialized(),
            threads=torch.get_num_threads(),
            interop=torch.get_num_interop_threads(),
            precision=torch.get_float32_matmul_precision(),
            torch=str(torch.__version__),
            source_packet=meta["sha256"],
            producer=prod["ref"],
            start_epoch_ns=start,
            end_epoch_ns=time_ns(),
            load_ns=load_ns,
            snapshot_ns=clock() - t,
            snapshots=digests,
            private_file_sha=sha(path),
        ),
    )


def difference(root, a, b):
    x, y = [load(root / k / "identity.json") for k in (a, b)]
    require(
        x["source_packet"] == y["source_packet"] and x["snapshots"].keys() == y["snapshots"].keys(),
        "wrong comparison",
    )
    entries = {}
    with (
        np.load(root / a / "private-tensors.npz", allow_pickle=False) as one,
        np.load(root / b / "private-tensors.npz", allow_pickle=False) as two,
    ):
        require(set(one.files) == set(two.files) == set(x["snapshots"]), "missing/private tensor")
        for k in one.files:
            u = one[k]
            v = two[k]
            require(
                all(
                    meta["snapshots"][k] == dict(sha=fingerprint(arr), shape=list(arr.shape))
                    for meta, arr in [(x, u), (y, v)]
                ),
                "private evidence corrupted",
            )
            entries[k] = summary(u, v)
    return entries


def main(root):
    os.umask(0o077)
    no_card()
    start = clock()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(
        head == (root / "gated-ref").read_text().strip()
        and "101 passed" in (root / "qualification.txt").read_text().splitlines()[-1],
        "not frozen/gated",
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "ndm/memory_program",
            str(HERE),
            "tests/test_memory_program*.py",
        ],
        check=True,
    )
    dump(root / "driver.json", dict(pid=os.getpid(), ref=head, order=ORDER, card_s=0, started=time_ns()))
    for cell in ORDER:
        t = clock()
        with (root / (cell + ".console")).open("x") as log:
            subprocess.run(
                [sys.executable, "-m", MODULE, str(root), "--cell", cell],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=min(115, 720 - (clock() - start) / 1e9),
                stdin=subprocess.DEVNULL,
            )
        print(cell, round((clock() - t) / 1e9, 2), "s incl process setup and exit", flush=True)
    diff = {f"{a}_vs_{b}": difference(root, a, b) for a, b in PAIRS}
    dump(root / "tensor-differences.json", diff)
    dump(
        root / "complete.json",
        dict(
            wall_ns=clock() - start,
            targets=4096 * len(ORDER),
            updates=len(ORDER),
            end_epoch_ns=time_ns(),
            card_s=0,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", choices=ORDER)
    args = ap.parse_args()
    worker(args.root, args.cell) if args.cell else main(args.root)
