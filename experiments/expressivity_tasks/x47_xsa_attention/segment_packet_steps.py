"""Strict CPU update check after T6 release. Compare private arrays, publish only evidence."""

import argparse, hashlib, os, subprocess, sys
from pathlib import Path
from time import perf_counter_ns as clock, time_ns
import numpy as np
import torch
from ndm.memory_program import PackedMemoryPolicy
from ndm.memory_program.frozen_packet import FrozenPacket
from .cpu_repro_bench import fingerprint
from .frozen_packet_bench import CAP, SEED, canonical, no_card, step
from .memory_program_profile import training_data
from .segment_packet_bench import binding
from .serving_bench import CK, CK_SHA
from .serving_verify import HERE, fields, ledger, load, require
from .workbench_run import dump, sha
from .workbench_witness import digest_state

ORDER = ("retained-serial", "retained-segment", "replicated-segment", "replicated-serial")
MODULE = "experiments.expressivity_tasks.x47_xsa_attention.segment_packet_steps"


def producer(root):
    aa, good = ledger(root, 32768, work_key="total_frontiers", actor_key="producer_pid")
    require(len(good) == 1 and sum(x["upper"] for x in aa) <= 115, "no released producer")
    return good[0]


def differences(a, b, kind):
    if (
        a.dtype != np.float32
        or a.dtype != b.dtype
        or a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("unpaired tensors")
    x = a.reshape(-1)
    y = b.reshape(-1)
    e = np.abs(x.astype(np.float64) - y.astype(np.float64))
    mag = np.abs(x.astype(np.float64))
    m = float(mag.max())
    bits = int(np.count_nonzero(x.view(np.uint32) != y.view(np.uint32)))
    if kind == "grad":
        bad = int((e > 2e-4 + 2e-4 * m).sum())
    elif kind == "score":
        bad = int((e > 5e-5 + 2e-5 * mag).sum())
    elif kind == "param":
        bad = int((e > 2e-5 + 2e-5 * mag).sum())
    elif kind in ("adam.exp_avg", "adam.exp_avg_sq"):
        bad = int((e > 2e-6 + 2e-4 * mag).sum())
    elif kind == "adam.step":
        bad = bits
    else:
        raise ValueError("unregistered tensor check")
    return dict(elements=x.size, unequal=bits, violations=bad, max_abs=float(e.max()), ref_max=m)


def worker(root, cell):
    os.umask(0o077)
    no_card()
    start = time_ns()
    r = producer(root)
    idn = load(r / "identity.json")
    size, method = cell.split("-")
    run = root / "cpu" / cell
    run.mkdir(mode=0o700, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(SEED)
    require(
        idn["ref"] == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        and start >= int(fields(r / "released")["released_epoch"]) * 10**9
        and sha(CK) == CK_SHA,
        "producer binding",
    )
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    p = PackedMemoryPolicy(256).train()
    p.load_state_dict(ck["model"])
    require(digest_state(p) == idn["policy_sha"] == ck["policy_sha"], "wrong policy")
    ds = training_data(32)
    rows, ws, man, data = ds
    require(
        idn["data"] == dict(data, window129_sha=[hashlib.sha256(w).hexdigest() for w in ws])
        and man["mount"] == idn["mount"],
        "changed inputs",
    )
    meta = load(r / (cell + "-1.json"))
    b, spec = binding(size, method, rows, ws, data, idn)
    require(meta["model_spec"] == spec, "changed encoder")
    t = clock()
    artifact = FrozenPacket.load(r / (cell + "-1.f32"), b, meta["sha256"], byte_limit=CAP)
    load_ns = clock() - t
    snap = {}

    def capture(g, k, t):
        key = g + ":" + k
        require(t.device.type == "cpu" and key not in snap, "invalid private capture")
        snap[key] = t.detach().contiguous().numpy().copy()

    def prog(phase, k):
        dump(run / "progress.json", dict(phase=phase, scored=k, learner_pid=os.getpid()))

    x = step(run, "step", p, artifact.encoder, ds, prog, coupled=False, capture=capture, capture_scores=True)
    c = x["checks"]
    require(c["input_sha"] == meta["payload_sha"], "not the verified payload")
    if method == "serial":
        require(
            c == load(HERE / f"receipts/memory-program-t5/{size}-strict-A/step.json")["checks"],
            "serial failed T5 strict reference",
        )
    after = hashlib.sha256()
    mom = {}
    dig = {k: fingerprint(v) for k, v in snap.items()}
    for k, g in sorted(c["gradients"].items()):
        v = snap["param:" + k]
        after.update(f"{k}|torch.float32|{tuple(v.shape)}\0".encode())
        after.update(v.tobytes())
        require(g == dig.get("grad:" + k), "bad preclip snapshot")
        if g is not None:
            mom[k] = {s: dig["adam." + s + ":" + k] for s in ("step", "exp_avg", "exp_avg_sq")}
    require(
        after.hexdigest() == c["after"]
        and canonical(mom) == c["optimizer"]
        and all(dig["score:" + s] == c[s] for s in ("byte_vectors", "joint_scores", "nll", "objective")),
        "wrong numerical snapshots",
    )
    path = run / "private.npz"
    with path.open("xb") as f:
        np.savez(f, **snap)
    no_card()
    dump(
        run / "identity.json",
        dict(
            cell=cell,
            pid=os.getpid(),
            ref=idn["ref"],
            input_file=meta["sha256"],
            start_epoch_ns=start,
            end_epoch_ns=time_ns(),
            threads=torch.get_num_threads(),
            strict=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cuda_initialized=torch.cuda.is_initialized(),
            cuda_available=torch.cuda.is_available(),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            load_ns=load_ns,
            tensors={k: dict(shape=list(snap[k].shape), sha=v) for k, v in dig.items()},
        ),
    )


def main(root):
    os.umask(0o077)
    no_card()
    producer(root)
    dst = root / "cpu"
    dst.mkdir(exist_ok=False, mode=0o700)
    beg = clock()
    for cell in ORDER:
        with (dst / (cell + ".console")).open("x") as f:
            subprocess.run(
                [sys.executable, "-m", MODULE, str(root), "--cell", cell],
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        print(cell, "fresh post-release CPU step complete", flush=True)
    for size in ("retained", "replicated"):
        a, b = [dst / (size + "-" + s) for s in ("serial", "segment")]
        u, v = [load(p / "identity.json")["tensors"] for p in (a, b)]
        out = {}
        with (
            np.load(a / "private.npz", allow_pickle=False) as x,
            np.load(b / "private.npz", allow_pickle=False) as y,
        ):
            for k in sorted(set(x.files) & set(y.files)):
                xx, yy = x[k], y[k]
                require(
                    fingerprint(xx) == u[k]["sha"] and fingerprint(yy) == v[k]["sha"],
                    "private comparison file changed",
                )
                out[k] = differences(xx, yy, k.split(":")[0])
        dump(
            dst / (size + "-differences.json"),
            dict(serial_only=sorted(set(u) - set(v)), segment_only=sorted(set(v) - set(u)), tensors=out),
        )
    dump(
        dst / "complete.json",
        dict(targets=16384, steps=4, order=ORDER, wall_ns=clock() - beg, end_epoch_ns=time_ns()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", choices=ORDER)
    args = ap.parse_args()
    worker(args.root, args.cell) if args.cell else main(args.root)
