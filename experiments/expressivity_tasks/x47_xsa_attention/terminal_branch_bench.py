"""T10: exact terminal/certifying-parent m20 transport, including the real S and R frames."""

import argparse
from collections import Counter
from contextlib import closing
from functools import partial
from hashlib import sha256
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter_ns as clock, time, time_ns

import numpy as np
import torch
from ndm.memory_program import MaterializedBranches, PackedMemoryPolicy, Unavailable
from ndm.memory_program.m20 import TerminalBranches
from ndm.memory_program.frozen_packet import FrozenPacket
from .direct_control_bench import T4, add_hash, bind, fresh, setup
from .frozen_packet_bench import CAP, no_card, step, tensor_sha
from .frozen_route_verify import work
from .inband_route_check import delta
from .memory_program_profile import mount_batch, training_data
from .serving_bench import CK, CK_SHA
from .serving_verify import HERE, load, require
from .stream_packet_bench import binding
from .w3_data import Pages
from .workbench_run import dump, sha
from .workbench_witness import digest_state

ORDER = (
    "grouped-ref-A",
    "grouped-terminal-A",
    "grouped-terminal-B",
    "grouped-ref-B",
    "sampled-ref",
    "sampled-terminal",
)
GATES = 120
FD_NAME = "t10-private"
CAPTURE_LIMIT = 12 << 20


class WireMeter:
    """Meter the same IPC and reap for either checked workspace opener."""

    def __init__(self, **binding):
        self.wire = {k: Counter() for k in ("S", "R")}
        self.peak = 0
        super().__init__(**binding)

    def rpc(self, op, n, payload, size):
        tm = clock()
        result = super().rpc(op, n, payload, size)
        c = self.wire[op.decode("ascii")]
        c.update(
            frames=1, items=n, send_bytes=5 + len(payload), recv_bytes=len(result), wall_ns=clock() - tm
        )
        return result

    def close(self):
        try:
            if self.proc.poll() is None:
                lines = Path(f"/proc/{self.proc.pid}/status").read_text().splitlines()
                self.peak = int(next(v.split()[1] for v in lines if v.startswith("VmHWM:")))
        finally:
            super().close()

    def receipt(self):
        require(self.proc.returncode == 0 and self.peak > 0, "wire worker not reaped")
        return dict(
            pid=self.proc.pid,
            rc=self.proc.returncode,
            peak_kib=self.peak,
            handshake=16,
            rpc=self.wire,
            binary=self.exe_sha,
            mount=self.desc.version,
            bytes=16 + sum(v["send_bytes"] + v["recv_bytes"] for v in self.wire.values()),
        )


class WirePages(WireMeter, Pages):
    pass


def oracle(root):
    tic = clock()
    _, ws, man, _ = training_data(32)
    keys = tuple(
        dict.fromkeys(
            [b"", b"\x00", b"\x1e", b"\xff", b"\x00\x1e\xff", bytes(range(256))]
            + [w[:j] for w in ws[:8] for j in range(1, 17)]
        )
    )
    checks = []
    with closing(WirePages()) as ref, closing(WirePages()) as can:
        a = bind(ref, man)
        b = bind(can, man, TerminalBranches)
        for label in ("cold", "warm"):
            aa = a.histograms(keys)
            bb = b.histograms(keys)
            require(aa == bb, "different complete native query answers")
            h = sha256()
            cnt = Counter()
            for key, ans in zip(keys, bb, strict=True):
                add_hash(h, (key, ans))
                cnt[
                    "unavailable"
                    if isinstance(ans, Unavailable)
                    else "positive"
                    if any(ans[:256])
                    else "end_only"
                    if ans[256]
                    else "absent"
                ] += 1
            checks.append(dict(label=label, keys=len(keys), sha=h.hexdigest(), counts=cnt))
    dump(
        root / "oracle.json",
        dict(
            checks=checks,
            input_sha=[sha256(w[:-1]).hexdigest() for w in ws[:8]],
            lengths=Counter(map(len, keys)),
            reference=ref.receipt(),
            candidate=can.receipt(),
            wall_ns=clock() - tic,
        ),
    )


def sampled(root, name, fd, bank_type, factory):
    ref = setup(root, gate_count=GATES)
    torch.manual_seed(7809)
    tic = clock()
    start = time_ns()
    run = root / name
    prod = load(T4 / "identity.json")
    meta = load(T4 / "retained-packet.json")
    require(sha(CK) == CK_SHA, "wrong P2")
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    p = PackedMemoryPolicy(256).train()
    p.load_state_dict(ck["model"])
    require(digest_state(p) == ck["policy_sha"] == prod["policy_sha"], "changed policy")
    ds = training_data(32)
    rows, ws, man, data = ds
    b, spec = binding("retained", rows, ws, data, prod)
    require(
        spec == meta["model_spec"]
        and prod["data"] == dict(data, window129_sha=[sha256(w).hexdigest() for w in ws])
        and (man["mount"], man["exe_sha"]) == (prod["mount"], prod["branch_exe_sha"]),
        "wrong sampled TRAIN/features/mount",
    )
    packet = FrozenPacket.load(
        Path("output/memory-program-t4/attempt1/retained.f32"), b, meta["sha256"], byte_limit=CAP
    )
    snap = {}

    def capture(kind, k, v):
        snap[kind + ":" + k] = v.detach().clone()

    def progress(phase, k):
        dump(run / "progress.json", dict(pid=os.getpid(), phase=phase, targets=k))

    s = step(
        run,
        "step",
        p,
        packet.encoder,
        ds,
        progress,
        coupled=False,
        capture=capture,
        capture_scores=True,
        mount=partial(mount_batch, bank_type=bank_type, pages_factory=factory),
    )
    require(
        s["targets"] == 4096
        and s["optimizer_steps"] == 1
        and s["mode"] == "fresh_sample"
        and not s["coupled"],
        "not a fresh sampled step",
    )
    life = clock() - tic
    require(sum(v.numel() * v.element_size() for v in snap.values()) < CAPTURE_LIMIT, "diagnostic size")
    with os.fdopen(os.dup(fd), "wb") as f:
        np.savez(f, **{k: v.numpy() for k, v in snap.items()})
    no_card()
    dump(
        run / "cell.json",
        dict(
            name=name,
            ref=ref,
            pid=os.getpid(),
            parent=os.getppid(),
            start_epoch_ns=start,
            end_epoch_ns=time_ns(),
            threads=torch.get_num_threads(),
            strict=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cuda_initialized=torch.cuda.is_initialized(),
            cuda_available=torch.cuda.is_available(),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            torch=str(torch.__version__),
            lifecycle_ns=life,
            wall_ns=s["times"]["wall_to_update_ns"],
            peak_main_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ipc_bytes=os.fstat(fd).st_size,
            targets=4096,
            steps=1,
            mount=man["mount"],
            binary=man["exe_sha"],
            packet=meta["sha256"],
            features=meta["payload_sha"],
            core_s=s["cpu_core_s"],
            seed=7809,
            nll=s["nll"],
            snapshots={k: dict(shape=list(v.shape), sha=tensor_sha(v)) for k, v in snap.items()},
        ),
    )


def worker(root, name, fd):
    bank_type = TerminalBranches if "terminal" in name else MaterializedBranches
    pages = []

    def factory():
        require(not pages, "opened two m20 processes")
        pages.append(WirePages())
        return pages[0]

    require(
        load(root / "driver.json")["pid"] == os.getppid()
        and not (root / name / "cell.json").exists()
        and os.readlink(f"/proc/self/fd/{fd}").startswith("/memfd:" + FD_NAME)
        and os.fstat(fd).st_size == 0,
        "unbound worker",
    )
    if name.startswith("grouped"):
        fresh(
            root,
            name,
            fd,
            gate_count=GATES,
            order=ORDER,
            pages_factory=factory,
            bank_type=bank_type,
            fd_name=FD_NAME,
        )
    else:
        sampled(root, name, fd, bank_type, factory)
    dump(root / name / "wire.json", pages[0].receipt())


def main(root):
    tic = clock()
    ref = setup(root, gate_count=GATES)
    require(time() + 485 < 1795531800 and not (root / "driver.json").exists(), "no time or reused run")
    dump(
        root / "driver.json",
        dict(pid=os.getpid(), ref=ref, start_epoch_ns=time_ns(), order=ORDER, card_s=0),
    )
    oracle(root)
    base = {}
    history = load(HERE / "receipts/memory-program-t9/grouped-A/cell.json")
    strict = load(HERE / "receipts/memory-program-t5/retained-strict-A/step.json")
    snap = load(HERE / "receipts/memory-program-t6/cpu/retained-serial/identity.json")["tensors"]
    for name in ORDER:
        run = root / name
        run.mkdir()
        t = clock()
        family = name.split("-")[0]
        with (
            os.fdopen(os.memfd_create(FD_NAME, os.MFD_CLOEXEC), "w+b") as f,
            (run / "console").open("wb") as out,
        ):
            proc = subprocess.run(
                [sys.executable, "-m", __spec__.name, str(root), "--cell", name, "--fd", str(f.fileno())],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                pass_fds=(f.fileno(),),
                timeout=max(1.0, 470 - (clock() - tic) / 1e9),
            )
            dump(run / "exit.json", dict(rc=proc.returncode, process_wall_ns=clock() - t))
            require(proc.returncode == 0, "cell failed: " + name)
            s = load(run / "cell.json")
            w = load(run / "wire.json")
            require(os.fstat(f.fileno()).st_size == s["ipc_bytes"] <= CAPTURE_LIMIT, "IPC size")
            f.seek(0)
            with np.load(f, allow_pickle=False) as z:
                require(set(z.files) == set(s["snapshots"]), "missing snapshot")
                xx = {k: torch.from_numpy(z[k]) for k in z.files}
        require(
            all(
                v.dtype == torch.float32
                and list(v.shape) == s["snapshots"][k]["shape"]
                and tensor_sha(v) == s["snapshots"][k]["sha"]
                for k, v in xx.items()
            ),
            "corrupt diagnostic transfer",
        )
        diff = None if family not in base else {k: delta(base[family][k], v) for k, v in xx.items()}
        if family == "grouped":
            logical = all(
                s[k] == history[k]
                for k in (
                    "per_document",
                    "init",
                    "after",
                    "work",
                    "depth",
                    "availability",
                    "local",
                    "nll",
                    "gate_mean",
                    "preclip_norm",
                )
            )
            exact = s["snapshots"] == history["snapshots"]
        else:
            x = load(run / "step.json")
            logical = work(x) == work(strict)
            exact = s["snapshots"] == snap and x["checks"] == strict["checks"]
        dump(
            run / "comparison.json",
            dict(same_work=logical, same_historical_update=exact, vs_fresh_reference=diff),
        )
        require(
            logical and exact and (diff is None or all(v["unequal_bits"] == 0 for v in diff.values())),
            "transport changed work/learning",
        )
        base.setdefault(family, xx)
        print(name, round(s["wall_ns"] / 1e9, 3), "s,", w["bytes"], "m20 wire bytes", flush=True)
    no_card()
    dump(
        root / "complete.json",
        dict(
            pid=os.getpid(),
            end_epoch_ns=time_ns(),
            elapsed_ns=clock() - tic,
            targets=24576,
            direct_steps=4,
            program_steps=2,
            host_steps=0,
            card_s=0,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", choices=ORDER)
    ap.add_argument("--fd", type=int)
    a = ap.parse_args()
    worker(a.root, a.cell, a.fd) if a.cell else main(a.root)
