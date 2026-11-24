"""S14 off-card preparation: reproduce the T9 update once, then retain only its four parameter arrays."""

from __future__ import annotations

import argparse
from hashlib import sha256
import os
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter_ns as clock, time, time_ns

import numpy as np
import torch
from ndm.memory_program.direct import DirectTrust
from ndm.memory_program.m20 import TerminalBranches
from .direct_control_bench import PAYLOAD, fresh, setup
from .frozen_packet_bench import no_card, tensor_sha
from .serving_verify import HERE, load, require
from .terminal_branch_bench import WirePages
from .workbench_run import dump, sha
from .workbench_witness import digest_state

GATES = 129
NAME = "grouped-rebuild"
FD = "s14-transient"
HEAD = "direct-head.f32"
HEAD_BYTES = 17604 * 4
T9 = HERE / "receipts/memory-program-t9/grouped-A/cell.json"
T9_SHA = "69bcfe92a8c7d43c50c5c0af5c063508ca931b3c7f7e7ccff12f0ed3164e77f7"


def history():
    require(sha(T9) == T9_SHA, "changed independent T9 witness")
    return load(T9)


def worker(root, fd):
    native = []

    def pages():
        require(not native, "second m20 process")
        native.append(WirePages())
        return native[0]

    fresh(
        root,
        NAME,
        fd,
        gate_count=GATES,
        order=(NAME,),
        pages_factory=pages,
        bank_type=TerminalBranches,
        fd_name=FD,
    )
    dump(root / NAME / "wire.json", native[0].receipt())


def audit_step(root):
    """Receipt-only: no need for local weights, inputs, features, or GPU."""
    ref = (root / "gated-ref").read_text().strip()
    d = root / NAME
    h = history()
    s = load(d / "cell.json")
    driver = load(root / "driver.json")
    end = load(d / "exit.json")
    require(
        all(
            s[k] == h[k]
            for k in (
                "snapshots",
                "per_document",
                "parameters",
                "init",
                "after",
                "nll",
                "preclip_norm",
                "local",
                "depth",
                "availability",
                "gate_mean",
                "work",
                "trust_calls",
                "host_calls",
                "targets",
                "steps",
                "binary",
                "mount",
                "packet",
                "features",
                "seed",
                "torch",
            )
        )
        and s["ref"] == driver["ref"] == ref
        and s["parent"] == driver["pid"] != s["pid"]
        and s["strict"]
        and not s["warn_only"]
        and s["threads"] == 4
        and s["visible"] == ""
        and not s["cuda_initialized"]
        and not s["cuda_available"]
        and s["name"] == NAME
        and s["native_rc"] == end["rc"] == driver["card_s"] == 0
        and end["process_wall_ns"] >= s["lifecycle_ns"] >= s["wall_ns"] > 0
        and driver["start_epoch_ns"] <= s["start_epoch_ns"] < s["end_epoch_ns"]
        and load(d / "progress.json") == dict(pid=s["pid"], phase="updated", targets=4096),
        "S14 requires the actual historical strict CPU update",
    )
    from .terminal_branch_verify import wire

    w = load(d / "wire.json")
    wr = wire(w, True, dict(mount=h["mount"], branch_exe_sha=h["binary"]))
    require(w["pid"] == s["native_pid"] and wr["r_rows"] == s["pages"]["rows"], "unbound replay worker")
    return s


def load_head(root):
    s = audit_step(root)
    meta = load(root / "head.json")
    # Read the exact bounded bytes once, from the same regular file we checked. No pickle.
    with os.fdopen(os.open(root / HEAD, os.O_RDONLY | os.O_NOFOLLOW), "rb") as f:
        st = os.fstat(f.fileno())
        require(
            stat.S_ISREG(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o600 and st.st_size == HEAD_BYTES,
            "unexpected head file",
        )
        raw = f.read(HEAD_BYTES + 1)
    require(
        len(raw) == meta["bytes"] == HEAD_BYTES
        and meta["sha"] == sha256(raw).hexdigest()
        and meta["after"] == s["after"]
        and meta["ref"] == s["ref"]
        and meta["history"] == T9_SHA,
        "unbound head payload",
    )
    a = np.frombuffer(raw, dtype="<f4")
    with torch.device("cpu"):
        head = DirectTrust(256).eval().requires_grad_(False)
    pos = 0
    for name, p in sorted(head.named_parameters()):
        v = torch.from_numpy(a[pos : pos + p.numel()].reshape(p.shape).copy())
        require(tensor_sha(v) == s["snapshots"]["param:" + name]["sha"], "wrong T9 parameter")
        p.copy_(v)
        pos += v.numel()
    require(pos == len(a) and digest_state(head) == s["after"], "wrong rebuilt model")
    return head, meta


def rebuild(root):
    start = clock()
    ref = setup(root, gate_count=GATES)
    require(
        time() + 190 < 1795531800
        and all(not (root / p).exists() for p in ("driver.json", NAME, HEAD, "head.json")),
        "reused CPU job or too late",
    )
    d = root / NAME
    d.mkdir()
    dump(root / "driver.json", dict(ref=ref, pid=os.getpid(), start_epoch_ns=time_ns(), card_s=0))
    with os.fdopen(os.memfd_create(FD, os.MFD_CLOEXEC), "w+b") as fd, (d / "console").open("xb") as log:
        t = clock()
        pr = subprocess.run(
            [sys.executable, "-m", __spec__.name, "worker", str(root), "--fd", str(fd.fileno())],
            pass_fds=(fd.fileno(),),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=170,
        )
        dump(d / "exit.json", dict(rc=pr.returncode, process_wall_ns=clock() - t))
        require(pr.returncode == 0, "CPU head replay stopped")
        s = audit_step(root)
        require(0 < os.fstat(fd.fileno()).st_size == s["ipc_bytes"] <= PAYLOAD, "IPC size")
        fd.seek(0)
        with np.load(fd, allow_pickle=False) as z:
            require(set(z.files) == set(s["snapshots"]), "incomplete replay transfer")
            pieces = []
            for k in sorted(z.files):
                v = torch.from_numpy(z[k])
                require(
                    v.dtype == torch.float32
                    and list(v.shape) == s["snapshots"][k]["shape"]
                    and tensor_sha(v) == s["snapshots"][k]["sha"],
                    "wrong replay transfer",
                )
                if k.startswith("param:"):
                    pieces.append(v.numpy().astype("<f4", copy=False).tobytes())
    raw = b"".join(pieces)
    require(len(raw) == HEAD_BYTES, "wrong head-only payload size")
    with NamedTemporaryFile(dir=root) as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
        os.link(f.name, root / HEAD)
    dump(
        root / "head.json",
        dict(
            ref=ref,
            after=s["after"],
            bytes=len(raw),
            sha=sha256(raw).hexdigest(),
            history=T9_SHA,
            elapsed_ns=clock() - start,
            published_epoch_ns=time_ns(),
            card_s=0,
        ),
    )
    _, meta = load_head(root)
    no_card()
    print("Verified one actual strict direct update, sealed T9 post-step head", meta, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("rebuild", "worker", "check"))
    p.add_argument("root", type=Path)
    p.add_argument("--fd", type=int)
    a = p.parse_args()
    if a.action == "worker":
        worker(a.root, a.fd)
    elif a.action == "rebuild":
        rebuild(a.root)
    else:
        no_card()
        load_head(a.root)
        print("Private serving head validated off card.")
